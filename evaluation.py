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
from scipy.spatial import ConvexHull
import torch
import matplotlib
            
matplotlib.use('TkAgg')
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


predict_data='/home/yishun/simulation/test/paper3/AnACorLKA/tomopy/test_19835_real/19835_real_raw_128_rs_2222b_2_all3/after_sam_rs_new_crrun/19835_real_refine_pred_mask.npy'
gt='/mnt/data/yishun/real_data/19485/test_.npy'
predict_data='/home/yishun/simulation/test/paper3/AnACorLKA/tomopy/test_19835_real/19835_real_raw_128_rs_2222b_2_onlysyn3/after_sam_rs_new_crrun/19835_real_refine_pred_mask.npy'
cutoff=700
# predict_data='/home/yishun/simulation/test/paper3/AnACorLKA/tomopy/test_21116_real/21116_real_raw_128_rs_2222b_2_all3/after_sam_rs_new_crrun/21116_real_refine_pred_mask.npy'
# predict_data='/home/yishun/simulation/test/paper3/AnACorLKA/tomopy/test_21116_real/21116_real_raw_128_rs_2222b_2_onlysyn3/after_sam_rs_new_crrun/21116_real_refine_pred_mask.npy'
# gt='/mnt/data/yishun/real_data/21116/21116_new_.npy'
# cutoff=950
gt_np=np.load(gt)
predict_data_np=np.load(predict_data)
gt_np=gt_np[:cutoff]
predict_data_np=predict_data_np[:cutoff]
# fig, ax = plt.subplots(1, 2, figsize=(10, 10))
# ax[0].imshow(predict_data_np[450, :, :])
# ax[1].imshow(gt_np[450, :, :])
# ax[0].set_title("Predicted")
# ax[1].set_title("Ground Truth")
# plt.show()
# pdb.set_trace()

gt=torch.Tensor(gt_np)
predict_data=torch.Tensor(predict_data_np)
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

num_classes=4
# volume_batch = volume_batch.to(torch.uint8)
predict_data =torch.unsqueeze(predict_data, 0)
gt = torch.unsqueeze(gt, 0)



one_hot_pred = label2mask(predict_data, num_classes=num_classes)
one_hot_gt = label2mask(gt, num_classes=num_classes)
pred_class = torch.argmax(one_hot_pred[0], dim=0)
target_class = torch.argmax(one_hot_gt[0], dim=0)
    
t1=time.time()
for j in range(len(one_hot_pred)):
    
    # pdb.set_trace()
    crystal_mean_sensitivity, crystal_overall_accuracy, crystal_f1_score, crystal_jaccard_index, crystal_f2_score, crystal_precision = calculate_metrics(one_hot_pred[j], one_hot_gt[j], _class=3,pred_class=pred_class,target_class=target_class)
    liquor_mean_sensitivity, liquor_overall_accuracy, liquor_f1_score, liquor_jaccard_index, liquor_f2_score, liquor_precision = calculate_metrics(one_hot_pred[j], one_hot_gt[j], _class=1,pred_class=pred_class,target_class=target_class)
    loop_mean_sensitivity, loop_overall_accuracy, loop_f1_score, loop_jaccard_index, loop_f2_score, loop_precision = calculate_metrics(one_hot_pred[j], one_hot_gt[j], _class=2,pred_class=pred_class,target_class=target_class)
    back_mean_sensitivity, back_overall_accuracy, back_f1_score, back_jaccard_index, back_f2_score, back_precision = calculate_metrics(one_hot_pred[j], one_hot_gt[j], _class=0,pred_class=pred_class,target_class=target_class)
    print(f'crystal_mean_sensitivity: {crystal_mean_sensitivity:.4f}, liquor_mean_sensitivity: {liquor_mean_sensitivity:.4f}, loop_mean_sensitivity: {loop_mean_sensitivity:.4f}, back_mean_sensitivity: {back_mean_sensitivity:.4f}')
    print(f'crystal_overall_accuracy: {crystal_overall_accuracy:.4f}, liquor_overall_accuracy: {liquor_overall_accuracy:.4f}, loop_overall_accuracy: {loop_overall_accuracy:.4f}, back_overall_accuracy: {back_overall_accuracy:.4f}')
    print(f'crystal_f1_score: {crystal_f1_score:.4f}, liquor_f1_score: {liquor_f1_score:.4f}, loop_f1_score: {loop_f1_score:.4f}, back_f1_score: {back_f1_score:.4f}')
    print(f'crystal_jaccard_index: {crystal_jaccard_index:.4f}, liquor_jaccard_index: {liquor_jaccard_index:.4f}, loop_jaccard_index: {loop_jaccard_index:.4f}, back_jaccard_index: {back_jaccard_index:.4f}')
    print(f'crystal_f2_score: {crystal_f2_score:.4f}, liquor_f2_score: {liquor_f2_score:.4f}, loop_f2_score: {loop_f2_score:.4f}, back_f2_score: {back_f2_score:.4f}')
    print(f'crystal_precision: {crystal_precision:.4f}, liquor_precision: {liquor_precision:.4f}, loop_precision: {loop_precision:.4f}, back_precision: {back_precision:.4f}')
    total_sensitivity += (crystal_mean_sensitivity + liquor_mean_sensitivity + loop_mean_sensitivity) / 3
    total_accuracy += (crystal_overall_accuracy + liquor_overall_accuracy + loop_overall_accuracy) / 3
    total_f1_score += (crystal_f1_score + liquor_f1_score + loop_f1_score) / 3
    total_jaccard_index += (crystal_jaccard_index + liquor_jaccard_index + loop_jaccard_index) / 3
    total_f2_score += (crystal_f2_score + liquor_f2_score + loop_f2_score) / 3
    total_precision += (crystal_precision + liquor_precision + loop_precision) / 3
    print(f'total_sensitivity: {total_sensitivity:.4f}, total_accuracy: {total_accuracy:.4f}, total_f1_score: {total_f1_score:.4f}, total_jaccard_index: {total_jaccard_index:.4f}, total_f2_score: {total_f2_score:.4f}, total_precision: {total_precision:.4f}')
t2=time.time()
print(f'time: {t2-t1}')

