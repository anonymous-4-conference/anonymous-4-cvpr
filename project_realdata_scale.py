import numpy as np
import cupy as cp
from cupy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
from utils_projection import model_loading,compare_images,rotate_model
import pdb
import matplotlib.pyplot as plt
import tqdm
import time
import cv2
import argparse
from utils_projection import sort_dataset,background_extraction,crop_to_fit,log_metadata
import os
from concurrent.futures import ProcessPoolExecutor
import random
import json
parser = argparse.ArgumentParser(prog="Dataset Projection",
                                    description="")

parser.add_argument(
    "--slices-dir",
    type=str,
    default='/mnt/data/yishun/real_data/tomopy/training/label/',
    help="The starting point of the batch",
)
parser.add_argument(
    "--background-dir",
    type=str,
    default='/mnt/data_smaller/yishun/simulation/background/',
    help="The ending point of the batch",
)
parser.add_argument(
    "--proj-dir",
    type=str,
    default='/mnt/data/yishun/simulation/8th_batch/projections_real_scale_thesis_final/',
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
    default='13295 21116 16846',
    help="The ending point of the batch",
)
global args
args = parser.parse_args()



def compute_propagator(coarse_projection, delta_z, lambda_xray, pixel_size, fresnel=True):
    M, N = coarse_projection.shape
    k = 2 * np.pi / lambda_xray
    
    fx = cp.fft.fftfreq(N, d=pixel_size)
    fy = cp.fft.fftfreq(M, d=pixel_size)
    FX, FY = cp.meshgrid(fx, fy)
    
    if fresnel:
        result = cp.exp(1j * 2 * np.pi / lambda_xray * delta_z) * cp.exp(-1j * np.pi * lambda_xray * delta_z * (FX**2 + FY**2))
    else:
        result = cp.exp(1j * 2 * np.pi / lambda_xray * delta_z) * cp.exp(1j * 2 * np.pi / lambda_xray * delta_z * cp.sqrt(1 - (FX * lambda_xray) ** 2 - (FY * lambda_xray) ** 2))
    
    return result

def compute_transmission_function(sample_slice, delta, beta, lambda_xray, pixel_size):
    k = 2 * np.pi / lambda_xray
    delta_slice = cp.zeros_like(sample_slice, dtype=cp.float32)
    beta_slice = cp.zeros_like(sample_slice, dtype=cp.float32)
    unique_values = np.unique(sample_slice)
    for value in unique_values:
        value = int(value)
        if value == 0:
            continue
        # pdb.set_trace()
        delta_slice[sample_slice == value] = delta.get(value, -1) * pixel_size
        beta_slice[sample_slice == value] = beta.get(value, -1) * pixel_size
    T = cp.exp(-k * (beta_slice + 1j * delta_slice))
    return T

def compute_absorption_function(sample_slice, coefficients_mapping, lambda_xray, pixel_size):
    beta_slice = cp.zeros_like(sample_slice, dtype=cp.float32)
    unique_values = cp.unique(sample_slice)
    for value in unique_values:
        value = int(value)
        if value == 0:
            continue
        
        beta_slice[sample_slice == value] = coefficients_mapping.get(value, -1) * pixel_size * 1e6
    T = cp.exp(-beta_slice)
    return T

def propagate_wavefield(u, propagator):
    u_ft = fft2(u)
    u_ft *= propagator
    u = ifft2(u_ft)
    return u


def simulate_xray_propagation_worker(angle, sample, delta, beta, lam, pixel_size, delta_z, z, background_img,save_pth,axis,):
    sample = rotate_model(sample, -angle)
    sample = cp.array(sample)
    y_non_zero, x_non_zero = np.where(np.sum(sample, axis=0) > 0)
    y_limits = (y_non_zero.min(), y_non_zero.max())
    x_limits = (x_non_zero.min(), x_non_zero.max())
    total_length = z / delta_z
    if axis == "y":
        npy_axis = 1
        limits = y_limits
    elif axis == "x":
        npy_axis = 2
        limits = x_limits
    else:
        print("axis should be either x or y")

    coarse_projection = np.sum(sample, axis=npy_axis)
    size_y = coarse_projection.shape[0]
    size_x = coarse_projection.shape[1]
    wavefield = cp.ones((size_y, size_x), dtype=cp.complex64)
    backg = (cv2.imread(background_img, 2)).astype(np.float32)/65535
    backg=crop_to_fit(backg, wavefield)
    background = cp.sqrt(cp.array(backg))
    wavefield_final = wavefield.copy()
    wavefield_final = background * wavefield_final
    propagator = compute_propagator(coarse_projection, delta_z, lam, pixel_size)
    free_space_propagator = compute_propagator(coarse_projection, z, lam, pixel_size)
    counter = 0

    for i in range(sample.shape[npy_axis]):
        
        if i < limits[0] or i > limits[1]:
            continue
        if axis == "y":
            sample_slice = sample[:, i, :]
        elif axis == "x":
            sample_slice = sample[:, :, i]
        else:
            print("axis should be either x or y")
        # pdb.set_trace()
        counter += 1
        T = compute_transmission_function(sample_slice, delta, beta, lam, pixel_size)        
        wavefield *= T
        # wavefield = propagate_wavefield(wavefield, propagator)
        wavefield_final *= T
        wavefield_final = propagate_wavefield(wavefield_final, propagator)
        # print(cp.abs(wavefield_final)**2)
        

    wavefield_0 = wavefield.get()
    wavefield_0 =np.abs(wavefield_0)**2 

    # wavefield_final_0 = wavefield_final.get()
    wavefield_final_0 =propagate_wavefield(wavefield_final, free_space_propagator).get()
    # wavefield_final = np.abs(wavefield_final.get() )**2*255
    wavefield_final_0 =np.abs(wavefield_final_0)**2
    
    # real_image_paths = f'/mnt/data/yishun/flat_field/21116/img_21116_rot_00000.tiff'
    # real_img= cv2.imread(real_image_paths,2)
    # real_img[real_img>1]=1  
    # wavefield_final_0[wavefield_final_0>1]=1
    # wavefield_final_0=(wavefield_final_0*255).astype(np.uint8)
    # real_img=(real_img*255).astype(np.uint8)
    # wavefield_0=(wavefield_0*255).astype(np.uint8)
    # yxshift = [0, 50]
    # wavefield_0=np.roll(wavefield_0, yxshift[1], axis=1)    
    # wavefield_0=np.roll(wavefield_0, yxshift[0], axis=0)
    # wavefield_final_0=np.roll(wavefield_final_0, yxshift[1], axis=1)    
    # wavefield_final_0=np.roll(wavefield_final_0, yxshift[0], axis=0)
    
    # fig, axes = plt.subplots(1, 4, figsize=(20, 12))
    # im1 = axes[0].imshow(wavefield_final_0, cmap='gray')
    # axes[0].set_title('Wavefield with recursive propagation')
    # # fig.colorbar(im1, ax=axes[0])
    # im2 = axes[1].imshow(wavefield_0, cmap='gray')
    # axes[1].set_title('Wavefield without wave propagation')
    # # fig.colorbar(im2, ax=axes[1])
    # img3=axes[2].imshow(real_img, cmap='gray')
    # axes[2].set_title('Real Image')
    # # fig.colorbar(img3, ax=axes[2])
    # im3 =axes[3].imshow((real_img-wavefield_final_0)/real_img,cmap='viridis')
    # fig.colorbar(im3, ax=axes[3])
    # plt.show()
    # ssim_score, mse_score, psnr_score = compare_images(wavefield_final_0, real_img)
    # print(f'Image {i}: SSIM = {ssim_score}, MSE = {mse_score}, PSNR = {psnr_score}')
    # ssim_score, mse_score, psnr_score = compare_images(wavefield_0, real_img)
    # print(f'Image {i}: SSIM = {ssim_score}, MSE = {mse_score}, PSNR = {psnr_score}')
    
    cv2.imwrite(save_pth.replace("syris","proj"), (wavefield_0).astype('float32') )
    cv2.imwrite(save_pth, (wavefield_final_0.astype('float32') ))
    
    return wavefield_0
    # wavefield_0 = wavefield.copy()
    # free_space_propagator = compute_propagator(coarse_projection, z, lam, pixel_size)
    # air_T = compute_transmission_function(cp.zeros_like(sample_slice), delta, beta, lam, pixel_size)
    # detector_wavefield = air_T * wavefield
    # detector_wavefield = propagate_wavefield(detector_wavefield, free_space_propagator)
    # intern_propagator = compute_propagator(coarse_projection, (limits[1] - limits[0]) * delta_z, lam, pixel_size)
    
    # wavefield = propagate_wavefield(wavefield, free_space_propagator)
    # wavefield_1 = propagate_wavefield(absorption_map, intern_propagator)
    
    # print(f"transmission function time: {time_t}")
    # print(f"propagation time: {time_p}")
    
    # yxshift = [0, 0]
    # # yxshift = [0, -8]
    # # yxshift = [0, 50]
    # syn_img_rotated = cp.abs(wavefield_0) ** 2
    
    # syn_img_rotated = cp.roll(syn_img_rotated, yxshift[1], axis=1)    
    # syn_img_rotated = cp.roll(syn_img_rotated, yxshift[0], axis=0)
    
    # absorption_map = cp.roll(absorption_map, yxshift[1], axis=1)  
    # absorption_map = cp.roll(absorption_map, yxshift[0], axis=0)    
    
    # wavefield_0 = cp.roll(wavefield_0, yxshift[1], axis=1)  
    # wavefield_0 = cp.roll(wavefield_0, yxshift[0], axis=0)    
    
    # wavefield_1 = cp.roll(detector_wavefield, yxshift[1], axis=1)  
    # wavefield_1 = cp.roll(detector_wavefield, yxshift[0], axis=0)    
    
    # pure_absorption_map = cp.roll(pure_absorption_map, yxshift[1], axis=1)
    # pure_absorption_map = cp.roll(pure_absorption_map, yxshift[0], axis=0)
    
    # real_image_paths = '/mnt/data_smaller/yishun/real_data/16115_4p5kev/TiffSaver_1/img_16115_rot_00000.tiff'
    # # real_image_paths = f'/mnt/data/yishun/flat_field/21116/img_21116_rot_00000.tiff'
    # real_img = cv2.imread(real_image_paths, 2)
    
    
    # syn_img_rotated = syn_img_rotated.get()
    # absorption_map = absorption_map.get()
    # wavefield_0 = wavefield_0.get()
    # wavefield_1 = wavefield_1.get()
    # pure_absorption_map = pure_absorption_map.get()
    # real_img[real_img > 1] = 1
    # diff = np.abs(real_img - syn_img_rotated) / real_img * 100
    # syn_img_rotated[syn_img_rotated > 1] = 1
    
    # ssim_score, mse_score, psnr_score = compare_images((np.abs(wavefield_0)**2 * 65535).astype(cp.uint16), (real_img * 65535).astype(np.uint16))
    # print(f"SSIM: {ssim_score}, MSE: {mse_score}, PSNR: {psnr_score}")
    # # Create figure
    # fig, axes = plt.subplots(2, 5, figsize=(20, 12))

    # # Subplot 1
    # im1 = axes[0, 0].imshow(np.abs(absorption_map)**2, cmap='gray')
    # axes[0, 0].set_title('Absorption map')
    # fig.colorbar(im1, ax=axes[0, 0])

    # # Subplot 2
    # im2 = axes[0, 1].imshow(np.abs(wavefield_0)**2, cmap='gray')
    # axes[0, 1].set_title('Wavefield 0 with recursive propagation')
    # fig.colorbar(im2, ax=axes[0, 1])

    # # Subplot 3
    # im3 = axes[0, 2].imshow(np.abs(wavefield_1)**2, cmap='gray')
    # axes[0, 2].set_title('Wavefield 1 with detector wavefield')
    # fig.colorbar(im3, ax=axes[0, 2])

    # # Subplot 4
    # im4 = axes[0, 3].imshow(pure_absorption_map, cmap='gray')
    # axes[0, 3].set_title('Pure Absorption map')
    # fig.colorbar(im4, ax=axes[0, 3])
   
    # im45 = axes[0, 4].imshow(real_img, cmap='gray')
    # axes[0, 4].set_title('Real Image')
    # fig.colorbar(im45, ax=axes[0, 4])
     
    # im5 = axes[1, 0].imshow((np.abs(absorption_map)**2 - real_img) / real_img, cmap='viridis')
    # axes[1, 0].set_title('Percentage Difference of \n Absorption map')
    # fig.colorbar(im5, ax=axes[1, 0])

    # # Subplot 6
    # im6 = axes[1, 1].imshow((np.abs(wavefield_0)**2 - real_img) / real_img, cmap='viridis')
    # axes[1, 1].set_title('Percentage Difference of \n Wavefield 0 with recursive propagation')
    # fig.colorbar(im6, ax=axes[1, 1])

    # # Subplot 7
    # im7 = axes[1, 2].imshow((np.abs(wavefield_1)**2 - real_img) / real_img, cmap='viridis')
    # axes[1, 2].set_title('Percentage Difference of \n Wavefield 1 with direct propagation')
    # fig.colorbar(im7, ax=axes[1, 2])

    # # Subplot 8
    # im8 = axes[1, 3].imshow((pure_absorption_map - real_img) / real_img, cmap='viridis')
    # axes[1, 3].set_title('Percentage Difference of \n pure_absorption_map  with direct propagation')
    # fig.colorbar(im8, ax=axes[1, 3])

    # im89 = axes[1, 4].imshow((np.abs(wavefield_1)**2 - np.abs(wavefield_0)**2) / (np.abs(wavefield_0)**2), cmap='viridis')
    # axes[1, 4].set_title('Percentage Difference of \n Wavefield 1 with direct propagation')
    # fig.colorbar(im89, ax=axes[1, 4])
    # plt.tight_layout()
    # plt.show()
    # mu = (4*np.pi* 5.60825e-7)/4.13e-10 /1e6
    # pdb.set_trace()

def parallel_simulation(angle_list,  model2, refractive_indices, betas, coefficients, lambda_xray, pixel_size, delta_z, z, background_list, axis,save_dir,num_processes=8):
    # scale=[5e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2]
    # scale=[1e-2,5e-2,1e-1,5e-1,1]
    # ref_indices=refractive_indices.copy()
    # for s in scale:
    #     ref_indices[1]=refractive_indices[1]*s
    #     ref_indices[3]=refractive_indices[3]*s
    #     ref_indices[2]=refractive_indices[2]*s  
    #     print(ref_indices)
    # simulate_xray_propagation_worker(angle_list[0], model2, refractive_indices, betas, lambda_xray, pixel_size, delta_z, z, background_list[0], os.path.join(save_dir, f'projection_syris_{str(0).zfill(4)}.tiff'),axis)
    # pdb.set_trace()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i, angle in enumerate(angle_list):
            background_img = background_list[i]
            save_pth = os.path.join(save_dir, f'projection_syris_{str(i).zfill(4)}.tiff')
           
            if os.path.exists(save_pth):
                continue
            tasks.append((angle, model2, refractive_indices, betas, lambda_xray, pixel_size, delta_z, z, background_img, save_pth,axis))

        futures = [executor.submit(simulate_xray_propagation_worker, *task) for task in tasks]
        for future in tqdm.tqdm(futures, desc=f"Processing {dataset}", total=len(angle_list)):
            future.result()  # Wait for all tasks to complete

if __name__ == '__main__':
    # slices='/home/yishun/simulation/test/crystal_8_10_2_3_25_50'
    # slices='/mnt/data_smaller/yishun/simulation/slices/crystal_3_8_3_2_0_25'
    # slices_dir='/mnt/data_smaller/yishun/real_data/tomopy/training/label/'
    # background_dirs='/mnt/data_smaller/yishun/simulation/background/'

    # proj_dir='/mnt/data/yishun/simulation/8th_batch/projections_real_scale_2/'
    slices_dir =args.slices_dir
    background_dirs=args.background_dir
    proj_dir=args.proj_dir

    dataset_list=[path for path in os.listdir(slices_dir) if os.path.isdir(os.path.join(slices_dir,path))]  
    t1 = time.time()
    # baseline_21116_coefficients={1: 0.015831078, 3: 0.01312, 2: 0.01171, 0: 0, 4: 0}
    # for key in baseline_21116_coefficients:
    #     baseline_21116_coefficients[key]=baseline_21116_coefficients[key]*( 3.01/3.53)**3
    # print(baseline_21116_coefficients)
    # pdb.set_trace()
    coefficients_mapping = {
        "13295": {1: 0.0101998, 3: 0.008067, 2: 0.00686, 0: 0, 4: 0}, #4kev
        "13076": {1: 0.007566823757243604, 3: 0.006271002372361257, 2: 0.00559706080642914, 0: 0.0, 4: 0.0}, #4.5kev
        "13304": {1: 0.02069, 3: 0.01319, 2: 0.00622, 0: 0.0, 4: 0.0}, #4kev
        "16000": {1: 0.007566823757243604, 3: 0.006271002372361257, 2: 0.00559706080642914, 0: 0.0, 4: 0.0}, #4.5kev
        "13284": {1: 0.009814899552230603, 3: 0.008134094350698387, 2: 0.007259927198679734, 0: 0.0, 4: 0.0}, #4kev
        "13668": {1: 0.011814899552230603, 3: 0.006134094350698387, 2: 0.007259927198679734, 0: 0.0, 4: 0.0}, #4kev
        "21116": {1: 0.015831078, 3: 0.01312, 2: 0.01171, 0: 0, 4: 0}, #3.5kev
        "19485": {1:  0.0127108, 3:  0.0100461, 2: 0.01171, 0: 0, 4: 0}, #3.5kev
        "19505": {1: 0.018558, 3: 0.015990, 2: 0.01724, 0: 0, 4: 0}, #3kev
        "19413": {1: 0.018593, 3: 0.0169 , 2: 0.017073, 0: 0, 4: 0}, #3kev
        "19224": {1: 0.00984, 3: 0.00627 , 2: 0.00776, 0: 0, 4: 0}, #4kev
        "16846": {1: 0.0078148, 3: 0.006458, 2: 0.00559706080642914, 0: 0, 4: 0}, #4.5kev
        "19835": {1: 0.021176017232711803, 3: 0.02043396017026068, 2: 0.02161307710141662, 0: 0, 4: 0}, #3kev  
        "20072": {1: 0.0174651026190351, 3: 0.018114363224687662, 2:0.016614023157962333, 0: 0, 4: 0}, #3kev
    }
    
    # randomly picking from
    refractive_indices_mapping = {
        "4.5keV": {0: 0, 1: 1.43e-5, 2: 1.524e-5, 3: 1.6417e-5}, #4.5kev
        "4keV": {0: 0, 1: 1.62e-5, 2: 1.73e-5, 3: 1.86e-5}, #4kev
        "3.5keV": {0: 0, 1: 2e-5, 2: 2.52e-5, 3: 3.565e-5},  #3.5kev
        "3keV": {0: 0, 1: 2.94e-5, 2: 3.44e-5, 3: 3.01e-5},  #3kev
        # "19485": {0: 0, 1: 2e-5, 2: 2.52e-5, 3: 3.565e-5},
        
        # "16846": {0: 0, 1: 2.88e-6, 2: 3.07e-6, 3: 3.3e-6},
        # "16115": {0: 0, 1: 0, 2: 1.524e-5, 3: 0},
    }
    lambda_xray_mapping = {
        "4keV": 3.01e-10,
        "3.5keV": 3.53e-10,
        "3keV": 3.53e-10,
        "4.5keV": 2.76e-10,
    }
    energy_mapping = {
        "13295": "4keV",    # 4 keV
        "13076": "4.5keV",  # 4.5 keV
        "13304": "4keV",    # 4 keV
        "13284": "4keV",    # 4 keV
        "13668": "4keV",    # 4 keV
        "21116": "3.5keV",  # 3.5 keV
        "19485": "3.5keV",  # 3.5 keV
        "19505": "3keV",    # 3 keV
        "19224": "4keV",    # 3 keV
        "19413": "3keV",    # 3 keV
        "16846": "4.5keV",  # 4.5 keV
        "16000": "4.5keV",  # 4.5 keV
        "19835": "3keV",    # 3 keV
        "20072": "3keV",    # 3 keV 
    }
    background_datalist = ["20454"] #,"20457","20465","20468","20516","20518"]
    delta_scale=[1e-2,5e-2,1e-1,5e-1 ] #,5e-2,1e-1]
    loop_mapping = {"3.5keV": 3.032e-7, "4keV": 1.7689e-7,"3keV": 5.608e-7, "4.5keV":  1.10127e-7}
    beta_mapping = {}
    for key, coefficients in coefficients_mapping.items():
        beta_mapping[key] = {k: (v * lambda_xray_mapping[energy_mapping[key]]) / (4 * np.pi)*1e6 for k, v in coefficients.items()}
    # print(beta_mapping)
    required_dataset=['13295','13076','16846','13284','13304','13668','16000','21116','19224','19485','19413']  #,'13295','16846','16115','19505']
    # required_dataset=args.dataset_list.split(' ')
    
    required_dataset=['13304'] #,'20506','20453']
    not_required_dataset=['14116','16115','16102','13313','19505']
    dataset_list.sort(key=sort_dataset) 

    random.seed(42)

    pdb.set_trace()
    for scale in delta_scale:
        results_dict = {}
        for dataset in dataset_list:
    
        
        
            # if dataset  in not_required_dataset:
                # continue
            # if int(dataset) < 19413:
            #     continue
            if dataset  not in required_dataset:
                continue
            # if '13295'  not in dataset:
            #     continue
            slices=os.path.join(slices_dir,dataset)
            axis = "y"
            phase_shift = True
            # dataset = slices
            
            energy = energy_mapping[dataset]
            pixel_size = 0.3e-6  # Pixel size in meters (x, y)
            delta_z = pixel_size  # Voxel size in meters (z-direction)
            z = 4.9e-3  # distance from the sample to the detector
            betas = beta_mapping[dataset].copy()
            betas[2]=loop_mapping[energy]
            # crystal_random_abs = random.choice([0.8,  1.2])  #random.uniform(0.8, 1.2)
            # if crystal_random_abs==1.2:
            #     liquor_random_abs =0.8
            # else:
            #     liquor_random_abs=1.2
            liquor_random_abs=1
            crystal_random_abs=1
            betas[1]=betas[1]*liquor_random_abs
            betas[3]=betas[3]*crystal_random_abs
            
            print('betas:',betas)
            

            liquor_random_refractive_indices = random.uniform(0.8, 1.2)

            crystal_random_refractive_indices =random.uniform(0.8, 1.2)
            # liquor_random_refractive_indices = 0.9

            # crystal_random_refractive_indices =1.1
            refractive_indices = refractive_indices_mapping[energy].copy()
            for key in refractive_indices:
                refractive_indices[key]=refractive_indices[key] * scale
            refractive_indices[1]=refractive_indices[1]*liquor_random_refractive_indices
            refractive_indices[3]=refractive_indices[3]*crystal_random_refractive_indices
    
            print("crystal_random_refractive_indices and liquor_random_refractive_indices:",crystal_random_refractive_indices,liquor_random_refractive_indices)
            print('refractive_indices:',refractive_indices)

       
            # pdb.set_trace()
            background = random.choice(background_datalist)
            background_dir=os.path.join(background_dirs,str(background))
            print('background_dir:',background_dir)

            lambda_xray = lambda_xray_mapping[energy]
        
            save_dir = os.path.join(proj_dir,os.path.basename(slices)).replace('slices','projection')
            
            basename =os.path.basename(save_dir)
            rootname = os.path.dirname(save_dir)
            save_dir =os.path.join(rootname,str(f'delta_scale_{scale}'),basename)
            log_save_dir = os.path.join(rootname,str(f'delta_scale_{scale}'))
            print('save_dir:',save_dir)
            os.makedirs(save_dir,exist_ok=True)

            if dataset not in results_dict:
                results_dict[dataset] = {}
            results_dict[dataset]['background'] = background
            results_dict[dataset]['coefficients'] = coefficients_mapping[dataset]
            results_dict[dataset]['refractive_indices'] = refractive_indices
            results_dict[dataset]['betas'] = betas
            results_dict[dataset]['energy'] = energy
            results_dict[dataset]['refractive_scale'] = scale


            with open(os.path.join(log_save_dir,'meta.json'), 'w') as f:
                json.dump(results_dict, f, indent=4)
            # pdb.set_trace()
            
            # continue    
            model2, t_loading = model_loading(slices, dataset)
            print(f"Time for loading model is {t_loading} seconds")
            background_img_list = background_extraction(background_dir)
            background_list =[os.path.join(background_dir, img) for img in background_img_list]
            angle_list=np.linspace(0,180,len(background_list))
            t1 = time.time()
            print(f"Starting simulation for {dataset}")
            parallel_simulation(angle_list,  model2, refractive_indices, betas, coefficients, lambda_xray, pixel_size, delta_z, z, background_list,axis, save_dir,num_processes=args.num_processes)
            t2 = time.time()
            log_metadata(log_save_dir, special_note, slices_dir, background_dir, pixel_size, betas, refractive_indices, slices, t1, t2)
            print("=====================================================")
            print(f"Time for your_dataset_name is {t2 - t1} seconds")
            print("=====================================================")
            