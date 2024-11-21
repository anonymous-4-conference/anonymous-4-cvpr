import numpy as np
import cupy as cp
from cupy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
from utils_projection import model_loading, compare_images, rotate_model
import pdb
import matplotlib.pyplot as plt
import tqdm
import time
import cv2
import argparse
from utils_projection import sort_dataset, background_extraction, crop_to_fit, log_metadata, get_delta_beta, energy2wavelength
from utils_compute_gpu2 import simulate_xray_propagation
from density_pdb import calculate_density_from_pdb,calculate_mother_liquor_density
import os
from concurrent.futures import ProcessPoolExecutor
import random
import json
parser = argparse.ArgumentParser(prog="Dataset Projection",
                                 description="")

parser.add_argument(
    "--slices-dir",
    type=str,
    help="The starting point of the batch",
)
parser.add_argument(
    "--background-dir",
    type=str,
    help="The ending point of the batch",
)
parser.add_argument(
    "--proj-dir",
    type=str,
    help="The ending point of the batch",
)
parser.add_argument(
    "--num_processes",
    type=int,
    default=8,
    help="The ending point of the batch",
)
parser.add_argument(
    "--dataset_list",
    type=str,
    help="The ending point of the batch",
)
global args
args = parser.parse_args()



if __name__ == '__main__':

    slices_dir = args.slices_dir
    background_dirs = args.background_dir
    proj_dir = args.proj_dir

    dataset_list = [path for path in os.listdir(
        slices_dir) if os.path.isdir(os.path.join(slices_dir, path))]
    t1 = time.time()
    background_datalist = ["20454"]
    delta_scale = [1e-2, 5e-2, 1e-1, 5e-1]

    required_dataset = ['13304']
    dataset_list.sort(key=sort_dataset)
    random.seed(42)
    
    """for 13304, the beta values are measured experimentally and as follows"""
    beta_mapping = {'13304': {1: 4.955838237719828e-07, 3: 3.1593768175700596e-07, 2: 1.4898653377775413e-07, 0: 0.0, 4: 0.0}}

    """common experiment parameters"""
    pixel_size = 0.3e-6  # Pixel size in meters (x, y)
    delta_z = pixel_size  # Voxel size in meters (z-direction)
    z = 4.9e-3  # distance from the sample to the detector
    energy = 4000  # in eV
    lambda_xray = energy2wavelength(energy)  # in angstroms
    pdbid='1RQW'
    
    crystal_material = 'C4H6O6'
    crystal_density = 1.43  # in g/cm³
    
    crystal_density,_=calculate_density_from_pdb(pdbid)
    crystal_beta, crystal_delta = get_delta_beta(
        crystal_material, energy, crystal_density)
    
    
    mother_liquor_material = 'C6H14O6'
    mother_liquor_density,_ =calculate_mother_liquor_density(pdbid)
    mother_liquor_beta, mother_liquor_delta = get_delta_beta(
        mother_liquor_material, energy, mother_liquor_density)

    loop_material = 'C22H10N2O5'
    loop_density = 1.43  # in g/cm³
    loop_beta, loop_delta = get_delta_beta(loop_material, energy, loop_density)

    # 1 is mother liquor, 2 is mounting loop, 3 is crystal
    refractive_indices = {
        1: mother_liquor_delta, 2: loop_delta, 3: crystal_delta}
    # 1 is mother liquor, 2 is mounting loop, 3 is crystal

    

    for dataset in dataset_list:

        if dataset not in required_dataset:
            continue

        slices = os.path.join(slices_dir, dataset)
        axis = "y"
        phase_shift = True



        liquor_random_refractive_indices = random.uniform(0.8, 1.2)

        crystal_random_refractive_indices = random.uniform(0.8, 1.2)
        refractive_indices[1] = refractive_indices[1] * liquor_random_refractive_indices
        refractive_indices[3] = refractive_indices[3] * crystal_random_refractive_indices

        # print("crystal_random_refractive_indices and liquor_random_refractive_indices:",
        #       crystal_random_refractive_indices, liquor_random_refractive_indices)
        # print('refractive_indices:', refractive_indices)

        # pdb.set_trace()
        background = random.choice(background_datalist)
        background_dir = os.path.join(background_dirs, str(background))
        print('background_dir:', background_dir)

        save_dir = os.path.join(proj_dir, os.path.basename(
            slices)).replace('slices', 'projection')

        
        betas = beta_mapping[dataset]
        print('betas:', betas)
        print('deltas:', refractive_indices)
        print('lambda_xray:', lambda_xray)
        basename = os.path.basename(save_dir)
        rootname = os.path.dirname(save_dir)
        save_dir = os.path.join(rootname, str(
            f'crystalseg_simulation'), basename)
        print('save_dir:', save_dir)
        os.makedirs(save_dir, exist_ok=True)




        # continue
        model2, t_loading = model_loading(slices, dataset)
        print(f"Time for loading model is {t_loading} seconds")
        background_img_list = background_extraction(background_dir)
        background_list = [os.path.join(background_dir, img)
                           for img in background_img_list]
        angle_list = np.linspace(0, 180, len(background_list))
        t1 = time.time()
        print(f"Starting simulation for {dataset}")
        raw_proj=[]
        raw_proj =simulate_xray_propagation(angle_list, model2, refractive_indices, betas, lambda_xray,
                                            pixel_size, delta_z, z, background_list, save_dir, axis)
        # if gpu is enoug:
        #     then xxx
        # else:
        #     for i in range(len(angle_list)):
        #         save_pth = os.path.join(   save_dir, f'projection_syris_{str(i).zfill(4)}.tiff')
        #         proj_img = simulate_xray_propagation_cupy(angle_list[i], model2, refractive_indices, betas, lambda_xray,
        #                                         pixel_size, delta_z, z, background_list[i], save_pth, axis)
        #         raw_proj.append(proj_img)

        #         pdb.set_trace()

        t2 = time.time()

        print("=====================================================")
        print(f"Time for your_dataset_name is {t2 - t1} seconds")
        print("=====================================================")
