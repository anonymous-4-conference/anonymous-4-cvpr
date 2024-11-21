import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
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
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, SAM2ImagePredictor
import h5py
import cv2
from utils import ramps, losses
from dataset import Tomodataset, RandomCrop, ToTensor, Resize
import pdb
from monai.losses.dice import DiceLoss, DiceFocalLoss
from monai.inferers import SlidingWindowInferer
from utils_valid import *
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
# validation_visualization, tensor_to_numpy, calculate_accuracy,output2mask,show_masks,show_mask,show_points,show_box,get_points,systematic_sampling,get_central_coordinates,kmeans_clustering,find_contour,points_outside_bbox
from monai.metrics import DiceMetric
from monai.visualize.utils import blend_images
from train_tomopy import create_model
import time
from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp
from functools import partial
from apply_sam2 import *
from scipy import stats as st
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
parser.add_argument("--save-dir", type=str, default="/mnt/data_smaller/yishun/dlka/", help="model_name")
parser.add_argument(
    "--max_iterations", type=int, default=10000, help="maximum epoch number to train"
)  # 6000
parser.add_argument("--batch_size", type=int, default=1, help="batch_size per gpu")
parser.add_argument(
    "--labeled_bs", type=int, default=1, help="labeled_batch_size per gpu"
)
parser.add_argument(
    "--region-size", type=int, default=96, help="patch size of the input"
)
parser.add_argument(
    "--resize-size", type=int, default=256, help="resize size of the input"
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


# train_data_path='/home/yishun/simulation/test/paper3/deformableLKA/dataset/'
train_data_path = args.train_data_path

snapshot_path = args.save_dir

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(","))
max_iterations = args.max_iterations
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
region_size = (args.region_size, args.region_size, args.region_size)  # 96x96x96 for Pancreas
T = 0.1
Good_student = 0  # 0: vnet 1:resnet
args.notes = args.notes + f""" 
                 Good_student: {Good_student} T: {T}
                num_classes: {num_classes}
                region_size: {region_size}
                max_iterations: {max_iterations}
                base_lr: {base_lr}
                T: {T}
                train_classes: {train_classes}
                """

def count_parameters(model):
    """
    Calculate the total number of trainable parameters in a PyTorch model.

    Parameters:
    - model (nn.Module): The PyTorch model to analyze.

    Returns:
    - int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
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

def run_multiprocessing(train_plot, pred_mask_original_plot, label_original_plot, iter_num, image_path, save_dir,cores=32):
    # mp.set_start_method('spawn')  
    num_slices = pred_mask_original_plot.shape[2]

    # Create a partial function with the common arguments
    func = partial(process_slice, 
                   train_plot=train_plot, 
                   pred_mask_original_plot=pred_mask_original_plot, 
                   label_original_plot=label_original_plot,
                   iter_num=iter_num, 
                   image_path=image_path, 
                   save_dir=save_dir)

    # Use multiprocessing Pool to parallelize the processing
    with mp.Pool(processes=cores) as pool:
        pool.map(func, range(num_slices))
        
def process_slice(i, train_plot, pred_mask_original_plot, label_original_plot, iter_num, image_path, save_dir):
    image = torch.Tensor(train_plot[:, :, i, :, :])
    out = torch.Tensor(pred_mask_original_plot[:, :, i, :, :])
    target = torch.Tensor(label_original_plot[:, :, i, :, :])

    validation_visualization(
        image,
        out,
        target,
        iter_num,
        image_path,
        i,
        save_dir=save_dir,
        train=False,
    )

def validate_epoch(
    model, validation_loader, writer, epoch_num, save_dir, region_size, num_classes,t1,save_name="validation"
):
    model.eval()
    single_dice_loss = DiceLoss(sigmoid=True)
    total_dice_loss = 0.0

    total_accuracy = 0.0
    total_crystal_dice = 0.0
    total_crystal_accuracy = 0.0
    total_sensitivity=0
    total_accuracy=0
    total_f1_score=0
    total_jaccard_index=0
    total_f2_score=0
    total_precision=0
    total_hausdorff_distance=0
    window_infer = SlidingWindowInferer(
        roi_size=region_size, sw_batch_size=1, overlap=0.5
    )
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


            original_shape = sampled_batch['original_shape']
            # print(f'validation iter {i_batch}')
            image_path = sampled_batch["image_path"]

            output = window_infer(volume_batch, model)
            dice_loss_1 = DiceFocalLoss(softmax=True)(output, volume_label)
            cross_entropy_loss = F.cross_entropy(output, volume_label.argmax(dim=1))
            # total_dice_loss += cross_entropy_loss
            # if i_batch==0:
            #     total_dice_2=dice_metric(y_pred=output, y=volume_label)
            # else:
            #     total_dice_2+=dice_metric(y_pred=output, y=volume_label)
            # dice_loss_1 = dice_loss(output, volume_label)
            total_dice_loss += dice_loss_1
            if num_classes==4:
                crystal_dice=single_dice_loss(output[:,3,:,:,:], volume_label[:,3,:,:,:])
                total_crystal_dice+=crystal_dice
            # total_dice_2+=dice_loss_2.squeeze(0)
            # if '21116' in image_path[0]:
            #     continue
            for j in range(len(output)):
                accuracy,overall_dice_loss = calculate_accuracy(output[j], volume_label[j])
                
                # total_accuracy += accuracy
                if num_classes==4:
                    crystal_accuracy,dice_loss_crystal=calculate_accuracy(output[j], volume_label[j],_class=3)
                    liquor_accuracy,dice_loss_liquor=calculate_accuracy(output[j], volume_label[j],_class=1)
                    loop_accuracy,dice_loss_loop=calculate_accuracy(output[j], volume_label[j],_class=2)
                    back_accuracy,dice_loss_back=calculate_accuracy(output[j], volume_label[j],_class=0)
                    print(f'dataset: {image_path[j]}')
                    print(f'crystal_accuracy: {crystal_accuracy*100:.2f}, liquor_accuracy: {liquor_accuracy*100:.2f}, loop_accuracy: {loop_accuracy*100:.2f}, back_accuracy: {back_accuracy*100:.2f}')
                    print(f'crystal_dice_loss: {dice_loss_crystal:.4f}, liquor_dice_loss: {dice_loss_liquor:.4f}, loop_dice_loss: {dice_loss_loop:.4f}, back_dice_loss: {dice_loss_back:.4f}')
                    print(f'cross_entropy_loss: {cross_entropy_loss:.4f}')
                    total_crystal_accuracy+=crystal_accuracy
            
            for j in range(len(output)):
                
                
                total_accuracy += accuracy
                if num_classes==4:
                    crystal_mean_sensitivity, crystal_overall_accuracy, crystal_f1_score, crystal_jaccard_index, crystal_f2_score, crystal_precision, crystal_hausdorff_distance=calculate_metrics(output[j], volume_label[j],_class=3)
                    liquor_mean_sensitivity, liquor_overall_accuracy, liquor_f1_score, liquor_jaccard_index, liquor_f2_score, liquor_precision, liquor_hausdorff_distance=calculate_metrics(output[j], volume_label[j],_class=1)
                    loop_mean_sensitivity, loop_overall_accuracy, loop_f1_score, loop_jaccard_index, loop_f2_score, loop_precision, loop_hausdorff_distance=calculate_metrics(output[j], volume_label[j],_class=2)
                    back_mean_sensitivity, back_overall_accuracy, back_f1_score, back_jaccard_index, back_f2_score, back_precision, back_hausdorff_distance=calculate_metrics(output[j], volume_label[j],_class=0)
                    print(f'dataset: {image_path[j]}')
                    print(f'crystal_mean_sensitivity: {crystal_mean_sensitivity:.4f}, liquor_mean_sensitivity: {liquor_mean_sensitivity:.4f}, loop_mean_sensitivity: {loop_mean_sensitivity:.4f}, back_mean_sensitivity: {back_mean_sensitivity:.4f}')
                    print(f'crystal_overall_accuracy: {crystal_overall_accuracy:.4f}, liquor_overall_accuracy: {liquor_overall_accuracy:.4f}, loop_overall_accuracy: {loop_overall_accuracy:.4f}, back_overall_accuracy: {back_overall_accuracy:.4f}')
                    print(f'crystal_f1_score: {crystal_f1_score:.4f}, liquor_f1_score: {liquor_f1_score:.4f}, loop_f1_score: {loop_f1_score:.4f}, back_f1_score: {back_f1_score:.4f}')
                    print(f'crystal_jaccard_index: {crystal_jaccard_index:.4f}, liquor_jaccard_index: {liquor_jaccard_index:.4f}, loop_jaccard_index: {loop_jaccard_index:.4f}, back_jaccard_index: {back_jaccard_index:.4f}')
                    print(f'crystal_f2_score: {crystal_f2_score:.4f}, liquor_f2_score: {liquor_f2_score:.4f}, loop_f2_score: {loop_f2_score:.4f}, back_f2_score: {back_f2_score:.4f}')
                    print(f'crystal_precision: {crystal_precision:.4f}, liquor_precision: {liquor_precision:.4f}, loop_precision: {loop_precision:.4f}, back_precision: {back_precision:.4f}')
                    print(f'crystal_hausdorff_distance: {crystal_hausdorff_distance:.4f}, liquor_hausdorff_distance: {liquor_hausdorff_distance:.4f}, loop_hausdorff_distance: {loop_hausdorff_distance:.4f}, back_hausdorff_distance: {back_hausdorff_distance:.4f}')
                    print(f'cross_entropy_loss: {cross_entropy_loss:.4f}')     
                    total_sensitivity+=(crystal_mean_sensitivity+liquor_mean_sensitivity+loop_mean_sensitivity)/3
                    total_accuracy+=(crystal_overall_accuracy+liquor_overall_accuracy+loop_overall_accuracy)/3
                    total_f1_score+=(crystal_f1_score+liquor_f1_score+loop_f1_score)/3
                    total_jaccard_index+=(crystal_jaccard_index+liquor_jaccard_index+loop_jaccard_index)/3
                    total_f2_score+=(crystal_f2_score+liquor_f2_score+loop_f2_score)/3
                    total_precision+=(crystal_precision+liquor_precision+loop_precision)/3
                    total_hausdorff_distance+=(crystal_hausdorff_distance+liquor_hausdorff_distance+loop_hausdorff_distance)/3
                         
            # continue
        # print(f'average sensitivity: {total_sensitivity/len(validation_loader)}')
        # print(f'average accuracy: {total_accuracy/len(validation_loader)}')
        # print(f'average f1_score: {total_f1_score/len(validation_loader)}')
        # print(f'average jaccard_index: {total_jaccard_index/len(validation_loader)}')
        # print(f'average f2_score: {total_f2_score/len(validation_loader)}')
        # print(f'average precision: {total_precision/len(validation_loader)}')
        # print(f'average hausdorff_distance: {total_hausdorff_distance/len(validation_loader)}')
        # pdb.set_trace()
            # continue
            b, c, h, w, d = output.shape
            li_fom_list = []
            cr_fom_list = []
            lo_fom_list = []
            for i in range(c):
                if i == 0:
                    continue
                for j in range(h):
                    slice = output[0, i, j, :, :].cpu().numpy()
                    label = volume_label[0, i, j, :, :].cpu().numpy()
                   
                    fom = pratt_fom(slice, label)
                    if i == 1:
                        li_fom_list.append(fom)
                    elif i == 2:
                        lo_fom_list.append(fom)
                    elif i == 3:
                        cr_fom_list.append(fom)
            cr_fom_array = np.array(cr_fom_list)
            cr_fom_array =cr_fom_array[cr_fom_array!=0]
            print(f'liquor: {np.mean(np.array(li_fom_list)):.4f}, loop: {np.mean(np.array(lo_fom_list)):.4f}, crystal: {np.mean(np.array(cr_fom_array)):.4f}')
            
            dataset=os.path.basename(image_path[0].split('.')[0])
            save_dir=f'./test_{dataset}/{dataset}_{save_name}/before_sam_rs_128'
            os.makedirs(save_dir,exist_ok=True)
            run_multiprocessing(volume_batch.detach().cpu(), output.detach().cpu(), volume_label.detach().cpu(), iter_num, image_path, save_dir)
            # pdb.set_trace()
            # 
            # for i in range(0, h, 10):

            #     if i < h:
            #         image = volume_batch[:, :, i, :, :]
            #         out = output[:, :, i, :, :]
            #         target = volume_label[:, :, i, :, :]

            #         validation_visualization(
            #             image,
            #             out,
            #             target,
            #             iter_num,
            #             image_path,
            #             i,
            #             save_dir=save_dir,
            #             train=False,
            #         )
            # continue

          
            h5f = h5py.File(image_path[0], 'r') #+"/mri_norm2.h5", 'r')
            train = h5f['train'][:]
            label = h5f['label'][:]
            assert train.shape == label.shape  
     
            pred_mask = output2mask(output).squeeze(0).cpu().numpy()
            pred_mask_original=Resize(train.shape).resize_image(pred_mask,order=0)
            label_original=Resize(train.shape).resize_image(label,order=0)
            pred_mask_original=pred_mask_original.astype(np.uint8)
            label_original=label_original.astype(np.uint8)
            t2 = time.time()
            print(f'model validation time: {t2-t1}')
            print(f'validation iter {i_batch}')
          
            
            save_dir=f'./test_{dataset}/{dataset}_{save_name}/before_sam_rs_full'
            os.makedirs(save_dir,exist_ok=True)

            file_name = os.path.join(save_dir,f'./{dataset}_pred.h5')
            with h5py.File(file_name, 'w') as h5_file:
                h5_file.create_dataset('train', data=pred_mask_original)
                if label_original is not None:
                    h5_file.create_dataset('label', data=label_original)
            np.save(f'{save_dir}/{dataset}_pred_mask.npy', pred_mask_original)
        

            train_plot=np.expand_dims(np.expand_dims(train,axis=0),axis=0  )    
            pred_mask_original_plot=np.expand_dims(np.expand_dims(pred_mask_original,axis=0),axis=0  )
            pred_mask_original_plot =mask2onehot(torch.Tensor(pred_mask_original_plot),num_classes)
            label_original_plot=np.expand_dims(np.expand_dims(label_original,axis=0),axis=0  )
            label_original_plot =mask2onehot(torch.Tensor(label_original_plot),num_classes)

            # for i in range(pred_mask_original_plot.shape[2]):
            #     # if '21116' in image_path[0] or  '20072' in image_path[0]:  
            #     #     continue
            #     try:
            #         process_slice(i, train_plot, pred_mask_original_plot, label_original_plot, iter_num, image_path, save_dir)
            #     except:
            #         print(f'error in slice {i} in dataset {dataset}')
            #         continue
            # run_multiprocessing(train_plot, pred_mask_original_plot, label_original_plot, iter_num, image_path, save_dir, cores=4)
            

            t1 = time.time()
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            sam2_checkpoint = "/home/yishun/simulation/test/paper3/segment-anything-2/checkpoints/sam2_hiera_large.pt"
            model_cfg = "./sam2_hiera_l.yaml"
            import matplotlib
            
            matplotlib.use('TkAgg')
            
            
            
            # pdb.set_trace()
            refine_pred_mask = pred_mask_original.copy()
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
            predictor = SAM2ImagePredictor(sam2_model)

            save_dir=f'./test_{dataset}/{dataset}_{save_name}/after_sam_rs_new'
            os.makedirs(save_dir,exist_ok=True)
            for i in tqdm(range(pred_mask_original.shape[0]),desc=f"Refine [{dataset}]"):
                # pdb.set_trace()
                # break
                
                # pred_slice = pred_mask_original[i]
                # pred_slice = pred_mask_original[385]
          
            # for i in [1050]:
                # i=117
                train_img = train[i]
                raw_img = cv2.cvtColor(train_img, cv2.COLOR_GRAY2RGB)
                plot_raw_img = train_img
                predictor.set_image(raw_img)
 
                new_img=raw_img.copy()
                new_img[pred_mask_original[i]==0]=0
                pred_cr_masks = np.zeros_like(pred_mask_original[i])
                pred_li_masks = np.zeros_like(pred_mask_original[i])
                pred_lo_masks = np.zeros_like(pred_mask_original[i])
                pred_back_masks = np.zeros_like(pred_mask_original[i])
            
                pred_cr_masks[pred_mask_original[i]==3]=1
                pred_li_masks[pred_mask_original[i]==1]=1
                pred_lo_masks[pred_mask_original[i]==2]=1
                pred_back_masks[pred_mask_original[i]==0]=1
                
                cr_input_points,cr_bbox= get_points(pred_mask_original[i],value=3)
                liquor_points,li_bbox= get_points(pred_mask_original[i],value=1,Erosion=True)
                loop_points,lo_bbox= get_points(label[i],value=2,Opening=True,kernel_size=(10,10))
                back_points,back_bbox= get_points(label[i],value=0)
                # pdb.set_trace()
                if (len(cr_input_points)<10 and len(loop_points)<20) or (len(cr_input_points)<10 and len(liquor_points)<10) :
                    continue    
           
                # detecting loop regions
                try:
                    lo_group_1_points, lo_group_2_points, lo_bbox_group_1, lo_bbox_group_2=seperate_loops(loop_points)
                except:
                    continue


                if lo_group_2_points is not None:
                    lo_masks_1,_,_= postprocess_sam(predictor,raw_img,lo_group_1_points,liquor_points,lo_bbox_group_1,bb_scale_factor=1.05,crystal=True,) #plot=True,
                    new_img[lo_masks_1[0]==1]=0   
                    lo_masks_2,_,_= postprocess_sam(predictor,raw_img,lo_group_2_points,liquor_points,lo_bbox_group_2,bb_scale_factor=1.05,crystal=True,) #plot=True,
                    new_img[lo_masks_2[0]==1]=0
                else:
                    lo_bbox_group_1=None
                    lo_masks_1,_,_= postprocess_sam(predictor,raw_img,lo_group_1_points,liquor_points,lo_bbox_group_1,bb_scale_factor=1.05) #plot=True,
                    new_img[lo_masks_1[0]==1]=0   
                    lo_masks_2=0
                    
                li_pixel_value =st.mode(raw_img[tuple(liquor_points.T)].flatten())[0]
                new_img[new_img==0]=li_pixel_value
                overall_li_masks,_,_= postprocess_sam(predictor,raw_img,liquor_points,back_points,li_bbox,in_number=10,bb_scale_factor=1.2,) #,plot=True
                
                # predictor.set_image(new_img)
                # if len(cr_input_points)<100:
                #     cr_masks_1=np.zeros_like(pred_cr_masks)
                #     cr_masks_1 = np.expand_dims(cr_masks_1,axis=0)
                # else:
                #     #
                #     cr_masks_1,_,_= postprocess_sam(predictor,new_img,cr_input_points,liquor_points,cr_bbox,in_number=10,bb_scale_factor=1,crystal=True,) #plot=True,
                # cr_masks_1 = cr_masks_1*3
                # pdb.set_trace()
                
                overall_li_masks = overall_li_masks*1
                lo_masks = (lo_masks_1+lo_masks_2)*2
                diff = np.abs(overall_li_masks*2 - lo_masks)
                diff_bbox=np.array(li_bbox) - np.array(lo_bbox)
                different_bbox=np.sqrt(np.sum(diff_bbox**2))
                
                cr_masks_1 = pred_cr_masks.copy()*3
                if len(cr_input_points)<30**3:
                    close_kernel=np.ones((10,10),np.uint8)
                elif len(cr_input_points)<30**3 and len(cr_input_points)<50**3:
                    close_kernel=np.ones((30,30),np.uint8)
                else:
                    close_kernel=np.ones((50,50),np.uint8)
                cr_masks_1 =cv2.morphologyEx(cr_masks_1, cv2.MORPH_CLOSE, close_kernel)
                # pdb.set_trace()
                if different_bbox<100 and np.count_nonzero(diff)<10000:
                    close_kernel_lo =np.ones((100,100),np.uint8)
                    open_kernel_li =np.ones((10,10),np.uint8)
                    open_kernel =np.ones((15,15),np.uint8)
                    lo_masks =  pred_lo_masks.copy()
                    # overall_li_masks = pred_li_masks.copy()
                    cr_masks_1=np.zeros_like(overall_li_masks)
                    
                    lo_masks =cv2.morphologyEx(pred_lo_masks, cv2.MORPH_CLOSE, close_kernel_lo)
                    # lo_masks =cv2.morphologyEx(lo_masks, cv2.MORPH_OPEN, open_kernel)
                    # overall_li_masks =cv2.morphologyEx(pred_li_masks, cv2.MORPH_OPEN, open_kernel_li)
                    # overall_li_masks =cv2.morphologyEx(overall_li_masks, cv2.MORPH_OPEN, open_kernel)
                    
                    # overall_li_masks = np.expand_dims(overall_li_masks,axis=0)
                    lo_masks = np.expand_dims(lo_masks,axis=0)*2
                    # cr_masks_1 = np.expand_dims(cr_masks_1,axis=0)*3
                if cr_masks_1.shape != lo_masks.shape:
                    cr_masks_1 = np.expand_dims(cr_masks_1,axis=0)
                try:
                    final_mask = np.maximum.reduce([cr_masks_1, overall_li_masks, lo_masks])
                except:
                    print(f'crystal: {cr_masks_1.shape}, liquor: {overall_li_masks.shape}, loop: {lo_masks.shape}')
                    pdb.set_trace()
                refine_pred_mask[i] = final_mask
                fig, ax = plt.subplots(2, 2, figsize=(10, 10))
                ax[0][0].imshow(plot_raw_img)
                ax[0][0].set_title('raw image')
                ax[0][1].set_title('ground truth')
                ax[0][1].imshow(label_original[i])
                ax[1][0].imshow(pred_mask_original[i])
                ax[1][0].set_title('before SAM')
                ax[1][1].imshow(final_mask[0])
                ax[1][1].set_title('final mask after SAM')

                
                fig.suptitle(f'Slice {i}', fontsize=16)
                # pdb.set_trace()
                # plt.show()
                plt.savefig(f'{save_dir}/slice_{str(i).zfill(4)}.png')
                # final_mask_plot =mask2onehot(torch.Tensor(np.expand_dims(np.expand_dims(final_mask,axis=0),axis=0  )),num_classes)

                # validation_visualization(torch.Tensor(train_plot[:, :, i, :, :]), final_mask_plot[:, :, 0, :, :], torch.Tensor(label_original_plot[:, :, i, :, :]), iter_num, image_path, i, save_dir=save_dir, train=False)
            # pdb.set_trace()

                
            np.save(f'{save_dir}/{dataset}_refine_pred_mask.npy', refine_pred_mask)
        # continue
        # refine_pred_mask = np.load(f'{save_dir}/{dataset}_refine_pred_mask_rs.npy')
        
        final_refine_pred_mask = mask2onehot(torch.Tensor(np.expand_dims(np.expand_dims(refine_pred_mask,axis=0),axis=0  )),num_classes)
        t2 = time.time()
        print(f'applying SAM time: {t2-t1}')
        for j in range(len(final_refine_pred_mask)):
            accuracy,overall_dice_loss = calculate_accuracy(final_refine_pred_mask[j], label_original_plot[j])
            
            total_accuracy += accuracy
            if num_classes==4:
                crystal_accuracy,dice_loss_crystal=calculate_accuracy(final_refine_pred_mask[j], label_original_plot[j],_class=3)
                liquor_accuracy,dice_loss_liquor=calculate_accuracy(final_refine_pred_mask[j], label_original_plot[j],_class=1)
                loop_accuracy,dice_loss_loop=calculate_accuracy(final_refine_pred_mask[j], label_original_plot[j],_class=2)
                back_accuracy,dice_loss_back=calculate_accuracy(final_refine_pred_mask[j], label_original_plot[j],_class=0)
                log_file_path = os.path.join(save_dir, "validation_results.txt")
                with open(log_file_path, "a") as log_file:
                    log_file.write(f'dataset: {image_path[0]}\n')
                    log_file.write(f'crystal_accuracy: {crystal_accuracy*100:.2f}, liquor_accuracy: {liquor_accuracy*100:.2f}, loop_accuracy: {loop_accuracy*100:.2f}, back_accuracy: {back_accuracy*100:.2f}\n')
                    log_file.write(f'crystal_dice_loss: {dice_loss_crystal:.4f}, liquor_dice_loss: {dice_loss_liquor:.4f}, loop_dice_loss: {dice_loss_loop:.4f}, back_dice_loss: {dice_loss_back:.4f}\n')
                    log_file.write(f'cross_entropy_loss: {cross_entropy_loss:.4f}\n')
                total_crystal_accuracy+=crystal_accuracy
           
        pdb.set_trace()    
    avg_dice_loss = total_dice_loss / len(validation_loader) / batch_size
    avg_accuracy = total_accuracy / len(validation_loader) / batch_size
    
    avg_crystal_dice = total_crystal_dice / len(validation_loader) / batch_size
    avg_crystal_accuracy = total_crystal_accuracy / len(validation_loader) / batch_size
    # avg_dice_loss_2 = total_dice_2 / total_samples
    writer.add_scalar("validation/loss_dice", avg_dice_loss, epoch_num)
    writer.add_scalar("validation/accuracy", avg_accuracy, epoch_num)
    if num_classes==4:
        writer.add_scalar("validation/crystal_dice", avg_crystal_dice, epoch_num)
        writer.add_scalar("validation/crystal_accuracy", avg_crystal_accuracy, epoch_num)
    # pdb.set_trace()
    logging.info(
        "\n Validation Epoch: {} Average Dice Loss: {:.4f}".format(
            epoch_num, avg_dice_loss
        )
    )
    logging.info(   
        "Validation Epoch: {} Average Accuracy: {:.4f}".format(
            epoch_num, avg_accuracy
        )
    )
    logging.info(
        "Validation Epoch: {} Average Crystal Dice Loss: {:.4f}".format(
            epoch_num, avg_crystal_dice
        )
    )
    logging.info("Validation Epoch: {} Average Crystal Accuracy: {:.4f} \n ".format(
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
    # if os.path.exists(snapshot_path + '/code'):
    #    shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))
    t1 = time.time()
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))



    model_anacor =  create_model(region_size=region_size,resize_size=args.resize_size,depths=[2,2,2,2])
    model_save_path ='/mnt/data_smaller/yishun/anacornew/raw_128_rs_1b_2_moresyn/d_model_former_iter_53_0.1546.pth'
    model_save_path ='/mnt/data_smaller/yishun/anacornew/raw_128_rs_1b_3_moresyn_c/anacornet_iter_26_0.1459.pth'
    # model_save_path='/mnt/data_smaller/yishun/anacornew/raw_128_rs_1b_3_moresyn/anacornet_iter_0_0.6415.pth'
    model_save_path="/mnt/data_smaller/yishun/anacornew/raw_128_rs_2211b_1_moresyn/anacornet_iter_27_0.1426.pth"
    model_save_path = "/mnt/data_smaller/yishun/anacornew/raw_128_rs_2222b_2_all2/anacornet_iter_85_0.1909.pth"
    model_save_path = "/mnt/data_smaller/yishun/anacornew/raw_128_rs_2222b_2_onlysyn/anacornet_iter_32_0.2105.pth"
    model_save_path='/mnt/data_smaller/yishun/anacornew/raw_128_rs_2222b_1_onlyreal/anacornet_iter_36_0.5181.pth'
    model_save_path='/mnt/data_smaller/yishun/anacornew/raw_128_rs_2222b_2_onlyreal3/anacornet_iter_51_0.4894.pth'
    model_save_path='/mnt/data_smaller/yishun/anacornew/raw_128_rs_2222b_2_all3/anacornet_iter_32_0.1462.pth'
    # model_save_path='/mnt/data_smaller/yishun/anacornew/raw_128_rs_2222b_2_onlysyn3/anacornet_iter_14_0.1783.pth'
    save_name = model_save_path.split('/')[-2]
    save_vaild_name = f"valid_{save_name}"
    model_anacor.load_state_dict(torch.load(model_save_path))
    snapshot_path = os.path.join(snapshot_path, save_vaild_name)
    figure_save_path = os.path.join(snapshot_path, "figures") 
    os.makedirs(figure_save_path, exist_ok=True)

    # model_anacor.load_state_dict(torch.load('/mnt/data_smaller/yishun/dlka/new_128_real_@/d_lka_former_iter_5016_0.0312.pth'))
    # model_anacor.load_state_dict(torch.load('/mnt/data_smaller/yishun/dlka/new_128_real_2/d_lka_former_iter_2016_0.0712.pth'))
    # model_anacor.load_state_dict(torch.load('/mnt/data_smaller/yishun/dlka/new_128_onlyreal_3/d_lka_former_iter_2004_0.1127.pth'))
    # model_anacor.load_state_dict(torch.load('/mnt/data_smaller/yishun/dlka/new_512_all_0/d_lka_former_iter_5046_0.2750.pth'))
    # model_anacor.load_state_dict(torch.load('/mnt/data_smaller/yishun/dlka/new_512_all_0/d_lka_former_iter_6003_0.2321.pth'))
    train_classes ={'crystal':3, 'background':0, 'loop':2, 'liquor':1}
    input_interpolation=0

    db_validation = Tomodataset(
        base_dir=train_data_path,
        train_classes=train_classes,
        train=False,
        args=args,
        interpolation=input_interpolation,
        pre_interpolate=False,
        test_dataset=['21116_real','19835_real'], #['21116_real','13295_real','16846_real','20072_real'],#,'20467_sirt'], #['21116_sirt','13295_sirt','16846_sirt']  ,'20467_real' ,'19835_real','20453_real' '16846_real','13295_real','21116_real',
        common_transform=transforms.Compose(
            [
                Resize((args.resize_size, args.resize_size, args.resize_size)),
            ]
        ),
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

    # model_anacor.load_state_dict(torch.load('/home/yishun/simulation/test/paper3/deformableLKA/3D/tomopy/model/new_160_onlyreal2/d_lka_former_iter_4018_0.4385.pth'))
    writer = SummaryWriter(snapshot_path + "/log")
    iter_num = 0
    lr_ = base_lr


    vali_loss = validate_epoch(
                model_anacor,
                validation_loader,
                writer,
                iter_num,
                figure_save_path,
                region_size,
                num_classes,
                t1,
                save_name=save_name,
            )

