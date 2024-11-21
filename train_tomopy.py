import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from model.networks import AnACor_Net
from model.transformer_block import TransformerBlock_3D
from torchmetrics.classification import JaccardIndex
from utils import ramps, losses
import matplotlib.pyplot as plt
# from dataloaders.la_heart import LAHeart, RandomCrop, ToTensor, Resize
from dataset import Tomodataset, RandomCrop  # , ToTensor  #, Resize
import pdb
from monai.losses.dice import DiceLoss, DiceFocalLoss
from monai.inferers import SlidingWindowInferer
from utils_valid import validation_visualization, tensor_to_numpy, calculate_accuracy
from monai.metrics import DiceMetric

from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate,
    RandScaleIntensity,
    RandShiftIntensity,
    RandGaussianNoise,
    Resize,
    ToTensor,
    CastToType,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_data_path",
    type=str,
    default="/mnt/data/yishun/simulation/8th_batch/training/",
    help="Name of Experiment",
)  # todo change dataset path
parser.add_argument(
    "--exp", type=str, default="tomopy_test_arch", help="model_name"
)  # todo model name
parser.add_argument(
    "--save-dir",
    type=str,
    default="/mnt/data_smaller/yishun/anacor_net/",
    help="model_name",
)
parser.add_argument(
    "--max_epoch", type=int, default=100, help="maximum epoch number to train"
)  # 6000
parser.add_argument("--batch_size", type=int, default=1, help="batch_size per gpu")
parser.add_argument(
    "--labeled_bs", type=int, default=1, help="labeled_batch_size per gpu"
)
parser.add_argument(
    "--region-size", type=int, default=96, help="patch size of the input"
)
parser.add_argument(
    "--resize-size", type=int, default=128, help="resize size of the input"
)
parser.add_argument(
    "--base_lr", type=float, default=0.01, help="maximum epoch number to train"
)
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=4, help="num of classes")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--workers", type=int, default=8, help="num of workers to use")
### costs
parser.add_argument("--ema_decay", type=float, default=0.999, help="ema_decay")
parser.add_argument(
    "--consistency_type", type=str, default="mse", help="consistency_type"
)
parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
parser.add_argument(
    "--consistency_rampup", type=float, default=40.0, help="consistency_rampup"
)
parser.add_argument(
    "--notes",
    default="",
    type=str,
)
args = parser.parse_args()

train_data_path = args.train_data_path

snapshot_path = args.save_dir

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(","))
max_epoch = args.max_epoch
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = args.num_classes
if num_classes == 2:
    train_classes = {"crystal": 1, "background": 0}
elif num_classes == 4:
    train_classes = {"crystal": 3, "background": 0, "loop": 2, "liquor": 1}
elif num_classes == 3:
    train_classes = {"crystal": 2, "background": 0, "loop": 1}
else:
    raise ValueError("num_classes must be 2 or 4 or 3")
# train_classes ={'crystal':3, 'background':0, 'loop':2, 'liquor':1}
# train_classes ={'crystal':3, 'background':0,}
# train_classes ={'crystal':3, 'background':0,'loop':2}
region_size = (
    args.region_size,
    args.region_size,
    args.region_size,
)  # 96x96x96 for Pancreas
T = 0.1
Good_student = 0  # 0: vnet 1:resnet
args.notes = (
    args.notes
    + f""" 
                 Good_student: {Good_student} T: {T}
                num_classes: {num_classes}
                region_size: {region_size}
                max_epoch: {max_epoch}
                base_lr: {base_lr}
                T: {T}
                train_classes: {train_classes}
                """
)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def gateher_two_patch(vec):
    b, c, num = vec.shape
    cat_result = []
    for i in range(c - 1):
        temp_line = vec[:, i, :].unsqueeze(1)  # b 1 c
        star_index = i + 1
        rep_num = c - star_index
        repeat_line = temp_line.repeat(1, rep_num, 1)
        two_patch = vec[:, star_index:, :]
        temp_cat = torch.cat((repeat_line, two_patch), dim=2)
        cat_result.append(temp_cat)

    result = torch.cat(cat_result, dim=1)
    return result


def create_model(region_size=(96, 96, 96), resize_size=128, depths=[3, 3, 3, 3]):
    # Network definition
    patch_size = (2, 2, 2)
    # region_size=[96,96,96] # alreay apply large kernel size
    region_size = [128, 128, 128]
    input_size = [
        int(region_size[0] / patch_size[0])
        * int(region_size[1] / patch_size[1])
        * int(region_size[2] / patch_size[2]),
        int(region_size[0] / patch_size[0] / 2)
        * int(region_size[1] / patch_size[1] / 2)
        * int(region_size[2] / patch_size[2] / 2),
        int(region_size[0] / patch_size[0] / 4)
        * int(region_size[1] / patch_size[1] / 4)
        * int(region_size[2] / patch_size[2] / 4),
        int(region_size[0] / patch_size[0] / 8)
        * int(region_size[1] / patch_size[1] / 8)
        * int(region_size[2] / patch_size[2] / 8),
    ]

    logging.info(f"patch_size: {patch_size}")
    logging.info(f"input_size: {np.array(input_size,dtype=float)**(1/3)}")

    net = AnACor_Net(
        in_channels=1,
        out_channels=num_classes,
        img_size=list(region_size),
        feature_size=16,  # this the dimensio of the first encoder
        hidden_size=256,  # this the dimension of the last encoder, feature_size*16=hidden_size
        patch_size=patch_size,
        input_size=input_size,
        trans_block=TransformerBlock_3D,
        do_ds=False,
        depths=depths,
    )
    model = net.cuda()
    return model


def validate_epoch(
    model,
    validation_loader,
    writer,
    epoch_num,
    save_dir,
    region_size,
    num_classes,
    plot=True,
):
    model.eval()
    single_dice_loss = DiceLoss(sigmoid=True)
    total_dice_loss = 0.0

    total_accuracy = 0.0
    total_crystal_dice = 0.0
    total_crystal_accuracy = 0.0
    # window_infer = SlidingWindowInferer(
    #     roi_size=region_size, sw_batch_size=1, overlap=0.5
    # )
    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=True, ignore_empty=True
    )
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(
            enumerate(validation_loader),
            desc=f"Validation [{epoch_num}]",
            total=len(validation_loader),
        ):
            if i_batch == 0:
                batch_size = sampled_batch["train"].shape[0]
            volume_batch, volume_label = (
                sampled_batch["train"].cuda(),
                sampled_batch["onehot_label"].cuda(),
            )
            # print(f'validation iter {i_batch}')
            image_path = sampled_batch["image_path"]
            # output = window_infer(volume_batch, model)
          
            output = model(volume_batch)
            cross_entropy_loss = F.cross_entropy(output, volume_label.argmax(dim=1))
            # if i_batch==0:
            #     total_dice_2=dice_metric(y_pred=output, y=volume_label)
            # else:
            #     total_dice_2+=dice_metric(y_pred=output, y=volume_label)
            # dice_loss_1 = dice_loss(output, volume_label)

            if num_classes == 4:
                crystal_dice = single_dice_loss(
                    output[:, 3, :, :, :], volume_label[:, 3, :, :, :]
                )
                total_crystal_dice += crystal_dice
            # total_dice_2+=dice_loss_2.squeeze(0)
            for j in range(len(output)):
                crystal_accuracy, dice_loss_crystal = calculate_accuracy(
                    output[j], volume_label[j], _class=3
                )
                liquor_accuracy, dice_loss_liquor = calculate_accuracy(
                    output[j], volume_label[j], _class=1
                )
                loop_accuracy, dice_loss_loop = calculate_accuracy(
                    output[j], volume_label[j], _class=2
                )
                back_accuracy, dice_loss_back = calculate_accuracy(
                    output[j], volume_label[j], _class=0
                )
                logging.info(f"dataset: {image_path[j]}")
                logging.info(
                    f"crystal_accuracy: {crystal_accuracy*100:.2f}, liquor_accuracy: {liquor_accuracy*100:.2f}, loop_accuracy: {loop_accuracy*100:.2f}, back_accuracy: {back_accuracy*100:.2f}"
                )
                logging.info(
                    f"crystal_dice_loss: {dice_loss_crystal:.4f}, liquor_dice_loss: {dice_loss_liquor:.4f}, loop_dice_loss: {dice_loss_loop:.4f}, back_dice_loss: {dice_loss_back:.4f}"
                )
                logging.info(f"cross_entropy_loss: {cross_entropy_loss:.4f}")
                total_crystal_accuracy += crystal_accuracy
                total_accuracy += (
                    crystal_accuracy + liquor_accuracy + loop_accuracy
                ) / 3
                total_dice_loss += (
                    dice_loss_crystal + dice_loss_liquor + dice_loss_loop
                ) / 3
            # pdb.set_trace()
            # outputs = window_infer(volume_batch, model, pred_type="ddim_sample")

            # outputs = [post_pred(i) for i in decollate_batch(outputs)]
            # labels = [post_label(i) for i in decollate_batch(volume_label)]

            # dice_metric(y_pred=outputs, y=labels)

            # dice_value = dice_metric.aggregate().item()
            b, c, h, w, d = output.shape
            if plot:
                for i in range(0, h, 10):

                    if i < h:
                        image = volume_batch[:, :, i, :, :]
                        out = output[:, :, i, :, :]
                        target = volume_label[:, :, i, :, :]

                        validation_visualization(
                            image,
                            out,
                            target,
                            epoch_num,
                            image_path,
                            i,
                            save_dir=save_dir,
                            train=False,
                        )
    del volume_batch, volume_label, output
    torch.cuda.empty_cache()
    avg_dice_loss = total_dice_loss / len(validation_loader.dataset)
    avg_accuracy = total_accuracy / len(validation_loader.dataset)

    avg_crystal_dice = total_crystal_dice / len(validation_loader.dataset)
    avg_crystal_accuracy = total_crystal_accuracy / len(validation_loader.dataset)
    # avg_dice_loss_2 = total_dice_2 / total_samples
    writer.add_scalar("validation/loss_dice", avg_dice_loss, epoch_num)
    writer.add_scalar("validation/accuracy", avg_accuracy, epoch_num)
    if num_classes == 4:

        writer.add_scalar(
            "validation/crystal_dice",
            avg_crystal_dice.as_tensor().cpu().numpy(),
            epoch_num,
        )
        writer.add_scalar(
            "validation/crystal_accuracy", avg_crystal_accuracy, epoch_num
        )
    # pdb.set_trace()
    logging.info(
        "\n Validation Epoch: {} Average Dice Loss: {:.4f}".format(
            epoch_num, avg_dice_loss
        )
    )
    logging.info(
        "Validation Epoch: {} Average Accuracy: {:.4f}".format(epoch_num, avg_accuracy)
    )
    logging.info(
        "Validation Epoch: {} Average Crystal Dice Loss: {:.4f}".format(
            epoch_num, avg_crystal_dice
        )
    )
    logging.info(
        "Validation Epoch: {} Average Crystal Accuracy: {:.4f} \n ".format(
            epoch_num, avg_crystal_accuracy
        )
    )
    # logging.info('Validation Epoch: {} Average Dice Loss 2: {}'.format(epoch_num, avg_dice_loss_2))
    model.train()
    return avg_dice_loss


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    import time

    # print("sleep 4h")
    # time.sleep(4*3600)
    # if os.path.exists(snapshot_path + '/code'):
    #    shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    figure_save_path = os.path.join(snapshot_path, "figures")
    os.makedirs(figure_save_path, exist_ok=True)

    model_ = create_model(
        region_size=region_size, resize_size=args.resize_size, depths=[3, 3, 3, 3]
    )  # [2,2,2,2]
    # model_.load_state_dict(torch.load("/mnt/data_smaller/yishun/anacornew/raw_128_rs_1b_3_moresyn/anacornet_iter_25_0.1478.pth"))
    total_params = sum(p.numel() for p in model_.parameters() if p.requires_grad)
    logging.info(f"Total number of trainable parameters: {total_params}")
    train_classes = {"crystal": 3, "background": 0, "loop": 2, "liquor": 1}
    input_interpolation = (
        3  # 0: nearest, 1: linear, 2:  Quadratic spline, 3: cubic spline
    )
    # train_classes ={'crystal':3, 'background':0,}
    # train_classes ={'crystal':3, 'background':0,'loop':2}

    """RandElasticDeformation

    Significantly improves robustness to anatomical or structural variations in medical or natural datasets.
    Particularly helpful for learning spatial transformations.
    RandRotate

    Addresses rotational invariance, crucial for datasets where the orientation of objects is unpredictable.
    Boosts performance in datasets like medical scans, where structures can be viewed from multiple angles.
    RandFlip

    Simple yet effective for augmenting symmetrical and orientation-agnostic datasets (e.g., brain scans or volumetric cells).
    Low computational overhead with good performance gains.
    RandomCrop

    Essential for focusing the model on specific regions and reducing overfitting to global contexts.
    Improves learning of localized features.
    RandScaleIntensity

    Helps the model become invariant to brightness variations, enhancing its ability to generalize across different imaging conditions.
    RandShiftIntensity

    Similar to intensity scaling but affects contrast more directly. Improves generalization by simulating various imaging setups.
    RandGaussianNoise

    Adds noise robustness, useful for datasets with inherent noise. Its impact depends on how noisy the original dataset is.
    Resize

    Does not contribute directly to augmentation but ensures uniformity in the input size, which is critical for model compatibility.
    """
    dataset_transform = transforms.Compose(
        [
            ToTensor(),  CastToType(dtype=torch.float32),
            # RandRotate(
            #     range_x=(-30, -60),
            #     range_y=(-30, -60),
            #     range_z=(-30, -60),
            #     prob=1,
            #     lazy=True,
            # ),
            # Randomly scale intensity for brightness variation
            # RandScaleIntensity(factors=0.1, prob=0.5),
            # # # Randomly shift intensity for contrast variation
            # RandShiftIntensity(offsets=0.1, prob=0.5),
            # # # Add Gaussian noise for robustness to noise
            # RandGaussianNoise(mean=0.0, std=0.05, prob=0.5),
        ]
    )
    vali_transform = transforms.Compose(
        [ToTensor(), CastToType(dtype=torch.float32)]
    )  # , CastToType(dtype=torch.float32)
    # [
    #     Resize((args.resize_size, args.resize_size, args.resize_size),mode='nearest'),
    # ]

    db_train = Tomodataset(
        base_dir=train_data_path,
        train_classes=train_classes,
        train=True,
        args=args,
        interpolation=input_interpolation,
        common_transform=dataset_transform,
        sp_transform=None,
        # transforms.Compose(
        # [
        #     ToTensor(),
        # ]
        # ),
    )
    logging.info("train data size: {}".format(len(db_train)))
    logging.info("here is the detailed train data list")
    db_train[0]
    trainloader = DataLoader(
        db_train,
        batch_sampler=None,
        num_workers=args.workers,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
    )
    # transforms.Compose(
    #             [
    #                 Resize((args.resize_size, args.resize_size, args.resize_size)),
    #             ]
    #         ),

    db_validation = Tomodataset(
        base_dir=train_data_path,
        train_classes=train_classes,
        train=False,
        args=args,
        interpolation=input_interpolation,
        common_transform=vali_transform,
        sp_transform=transforms.Compose(
            [
                ToTensor(),
            ]
        ),
    )
    logging.info("validation data size: {}".format(len(db_validation)))
    logging.info("here is the detailed validation data list")
    db_validation.print_sample(logging)
    validation_loader = DataLoader(
        db_validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    window_infer = SlidingWindowInferer(
        roi_size=list(region_size), sw_batch_size=1, overlap=0.5
    )

    model_optimizer = optim.SGD(
        model_.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=50, eta_min=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model_optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )
    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} itertations per epoch".format(len(trainloader)))

    lr_ = base_lr

    model_.train()
    valid = False

    valid_iteration = 0
    valid_iteration_step = 1500

    lr_iteration = 10000
    lr_iteration_step = 10000

    train_iteration = 500
    train_iteration_step = 500
    best_vali_loss = 999
    iou_loss = JaccardIndex(task="multiclass", num_classes=num_classes).cuda()
    for epoch_num in range(max_epoch):
        time1 = time.time()
        epoch_loss = 0
        epoch_dice_loss = 0
        epoch_ce_loss = 0
        epoch_iou_loss = 0
        for i_batch, sampled_batch in tqdm(
            enumerate(trainloader),
            desc=f"Training [{epoch_num}/{max_epoch}]",
            total=len(trainloader),
        ):

            time2 = time.time()
            # class_weights = torch.tensor([0.4, 0.2, 0.2, 0.2]).cuda()

            image_path = sampled_batch["image_path"]
            basenames = [os.path.basename(path) for path in image_path]
            volume_batch1, volume_label1, onehot_label = (
                sampled_batch["train"],
                sampled_batch["label"],
                sampled_batch["onehot_label"],
            )
            # volume_batch1 =torch.unsqueeze(volume_batch1,1)
            # pdb.set_trace()
            # import matplotlib
            # matplotlib.use('TkAgg')
            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # ax[0].imshow(volume_batch1.cpu().numpy()[1, 0, 60, :, :])
            # ax[1].imshow(volume_label1.cpu().numpy()[1, 60, :, :])
            # plt.show()
            # pdb.set_trace()
            # volume_batch2, volume_label2 = sampled_batch[1]['image'], sampled_batch[1]['label']
            # Transfer to GPU
            model_input, model_label = volume_batch1.cuda(), onehot_label.cuda()
            #
            # Network forward

            model_logits = model_(model_input)

            ## calculate the su pervised loss
            model_loss_seg = F.cross_entropy(model_logits, model_label.float())
            # model_outputs_soft = F.softmax(model_outputs, dim=1)
            # model_loss_seg_dice = losses.dice_loss(model_outputs_soft[:, 1, :, :, :], model_label[:] == 1)
            del model_input
            torch.cuda.empty_cache()
            class_weights = torch.tensor([1, 1, 1, 3]).cuda()
            if num_classes == 4:
                model_loss_seg_dice = DiceFocalLoss(softmax=True, weight=class_weights)(
                    model_logits, model_label
                )
            else:
                model_loss_seg_dice = DiceFocalLoss(softmax=True)(
                    model_logits, model_label
                )
            loss_total = model_loss_seg + model_loss_seg_dice

            predictions = torch.argmax(F.softmax(model_logits, dim=1), dim=1)
            last_channel_pred = (
                predictions == (num_classes - 1)
            ).float()  # Convert to float for IoU calculation
            last_channel_label = (
                torch.argmax(model_label, dim=1) == (num_classes - 1)
            ).float()
            model_loss_seg_iou = 1 - iou_loss(last_channel_pred, last_channel_label)
            # model_loss_seg_iou = 1 - iou_loss( torch.argmax(F.softmax(model_logits, dim=1), dim=1),torch.argmax(model_label, dim=1) ) #F.softmax(model_logits, dim=1)
            loss_total = model_loss_seg + model_loss_seg_dice + model_loss_seg_iou

            # pdb.set_trace()
            model_optimizer.zero_grad()
            loss_total.backward()
            model_optimizer.step()

            current_iteration = (epoch_num + 1) * len(trainloader) * batch_size + (
                i_batch + 1
            ) * batch_size

            b, c, h, w, d = model_logits.shape

            # if epoch_num > 100:
            # pdb.set_trace()
            # if train_iteration < current_iteration:  # or "0004" in basenames[0]:
            if epoch_num % 5 == 0:
                for b in range(b):
                        image_path_name = basenames[b]
                        if "real" in image_path_name:
                            for i in range(0, h, 10):
                                if i <20 or i>100:
                                    continue    
                                image = volume_batch1[b, :, i, :, :]
                                output = model_logits[b, :, i, :, :]
                                target = model_label[b, :, i, :, :]
                                image = torch.unsqueeze(image, 0)
                                output = torch.unsqueeze(output, 0)
                                target = torch.unsqueeze(target, 0)
                                validation_visualization(
                                    image,
                                    output,
                                    target,
                                    epoch_num,
                                    [image_path[b]],
                                    i,
                                    save_dir=figure_save_path,
                                    train=True,
                                )
                            train_iteration += train_iteration_step
                            del image, output, target

            del model_label, model_logits
            torch.cuda.empty_cache()
            epoch_loss += loss_total.item()
            epoch_dice_loss += model_loss_seg_dice.item()
            epoch_iou_loss += model_loss_seg_iou.item()
            epoch_ce_loss += model_loss_seg.item()
            # if lr_iteration < current_iteration: # and epoch_num != 1:

            #     lr_iteration +=lr_iteration_step
            #     lr_ = lr_ * 0.1
            #     for param_group in model_optimizer.param_groups:
            #         param_group["lr"] = lr_
        # pdb.set_trace()
        # exit()
        epoch_loss /= len(trainloader.dataset)
        epoch_ce_loss /= len(trainloader.dataset)
        epoch_dice_loss /= len(trainloader.dataset)
        epoch_iou_loss /= len(trainloader)
        # current_lr = model_optimizer.param_groups[0]['lr']
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("lr", current_lr, epoch_num)
        writer.add_scalar("loss/total_loss", epoch_loss, epoch_num)
        writer.add_scalar("loss/model_loss_seg", epoch_ce_loss, epoch_num)
        writer.add_scalar("loss/model_loss_seg_dice", epoch_dice_loss, epoch_num)
        writer.add_scalar("loss/model_loss_seg_iou", epoch_iou_loss, epoch_num)
        # logging.info(f"current_iteration: {current_iteration}")
        logging.info(
            "iteration: %d Total loss : %f CE loss : %f Dice loss : %f Iou loss : %f"
            % (
                epoch_num,
                epoch_loss,
                epoch_ce_loss,
                epoch_dice_loss,
                epoch_iou_loss,
            )
        )
        time1 = time.time()
        # if valid_iteration < current_iteration : # and epoch_num != 1:
        #     valid = True
        #     valid_iteration +=valid_iteration_step
        #     print(f"valid_iteration: {valid_iteration}")
        # if epoch_num % 5 == 0:
        #     valid_plot=True
        # else:
        #     valid_plot=False
        valid_plot = True
        vali_loss = validate_epoch(
            model_,
            validation_loader,
            writer,
            epoch_num,
            figure_save_path,
            region_size,
            num_classes,
            plot=valid_plot,
        )

        if vali_loss < best_vali_loss:
            save_mode_path_model_net = os.path.join(
                snapshot_path,
                "anacornet_iter_" + str(epoch_num) + f"_{vali_loss:.4f}" + ".pth",
            )
            torch.save(model_.state_dict(), save_mode_path_model_net)
            logging.info("save model to {}".format(save_mode_path_model_net))
            best_vali_loss = vali_loss

        # valid = False
        scheduler.step(vali_loss)
        # scheduler.step()
        torch.cuda.empty_cache()
        if epoch_num >= max_epoch:
            break
    save_mode_path_model_net = os.path.join(
        snapshot_path, "anacornet_iter_" + str(max_epoch) + ".pth"
    )
    torch.save(model_.state_dict(), save_mode_path_model_net)
    logging.info("save model to {}".format(save_mode_path_model_net))

    writer.close()
