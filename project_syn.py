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
from utils_projection import sort_dataset,background_extraction,crop_to_fit
import os
from concurrent.futures import ProcessPoolExecutor
from project_realdata import simulate_xray_propagation_worker
from utils_projection import log_metadata
parser = argparse.ArgumentParser(prog="Dataset Projection",
                                    description="")

parser.add_argument(
    "--slices-dir",
    type=str,
    default='/mnt/data_smaller/yishun/simulation/slices',
    help="The starting point of the batch",
)
parser.add_argument(
    "--background-dir",
    type=str,
    default='/mnt/data_smaller/yishun/simulation/background/20457',
    help="The ending point of the batch",
)
parser.add_argument(
    "--proj-dir",
    type=str,
    default='/mnt/data_smaller/yishun/simulation/projections',
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

def parallel_simulation(angle_list,  model2, refractive_indices, betas, lambda_xray, pixel_size, delta_z, z, background_list, axis,save_dir,num_processes=16):
    # save_pth = os.path.join(save_dir, f'projection_syris_{str(0).zfill(4)}.tiff')
    # simulate_xray_propagation_worker(angle_list[0], model2, refractive_indices, betas, lambda_xray, pixel_size, delta_z, z, background_list[0], save_pth,axis)

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i, angle in enumerate(angle_list):
            background_img = background_list[i]
            save_pth = os.path.join(save_dir, f'projection_syris_{str(i).zfill(4)}.tiff')
           
            if os.path.exists(save_pth):
                continue
            tasks.append((angle, model2, refractive_indices, betas, lambda_xray, pixel_size, delta_z, z, background_img, save_pth,axis))

        futures = [executor.submit(simulate_xray_propagation_worker, *task) for task in tasks]
        for future in tqdm.tqdm(futures, desc=f"Processing {os.path.basename(save_dir)}", total=len(angle_list)):
            future.result()  # Wait for all tasks to complete

if __name__ == '__main__':
    # slices='/home/yishun/simulation/test/crystal_8_10_2_3_25_50'
    # slices='/mnt/data_smaller/yishun/simulation/slices/crystal_3_8_3_2_0_25'
    slices_dir='/mnt/data_smaller/yishun/simulation/4th_batch/slices/'
    background_dirs='/mnt/data_smaller/yishun/simulation/background/'
 
    proj_dir='/mnt/data/yishun/simulation/8th_batch/projections_syn3/'
    special_note='Use syn datasets, using 21116 parameters,scale down the refractive index by 5 scales,'
    # proj_dir='/mnt/data_smaller/yishun/simulation/4th_batch/projections_phase_detector/'
    # special_note='Adding Fernsel diffraction to enhance edge effect'
    # slices_dir=args.slices_dir
    # background_dir=args.background_dir
    # proj_dir=args.proj_dir
    dataset_list=[path for path in os.listdir(slices_dir) if os.path.isdir(os.path.join(slices_dir,path))]  
    t1 = time.time()
    delta_scale=[1e-2,5e-2,1e-1,5e-1,1]
    coefficients_mapping = {
        "13295": {1: 0.0071998, 3: 0.007467, 2: 0.00686, 0: 0, 4: 0},
        "21116": {1: 0.015831078, 3: 0.01312, 2: 0.01171, 0: 0, 4: 0},
        "16846": {1: 0.020187, 3: 0.019264, 2: 0.0186448, 0: 0, 4: 0},
        "16115": {1: 0, 3: 0, 2: 0.00534 , 0: 0, 4: 0},  # in meters {1: 0, 3: 0, 2: 0.00534e-6 1.11e-7, 0: 0, 4: 0}
    }
    background_mapping = {"13295": 20468, "21116": 20518, "16846": 20457}
    refractive_indices_mapping = {
        "13295": {0: 0, 1: 1.62e-6, 2: 1.73e-6, 3: 1.86e-6},
        "21116": {0: 0, 1: 2e-5, 2: 2.52e-5, 3: 3.565e-5},
        "16846": {0: 0, 1: 2.88e-6, 2: 3.07e-6, 3: 3.3e-6},
        "16115": {0: 0, 1: 0, 2: 1.524e-5, 3: 0},
    }
    lambda_xray_mapping = {
        "13295": 3.01e-10,
        "21116": 3.53e-10,
        "16846": 4.13e-10,
        "16115": 2.76e-10,
    }
    loop_mapping = {'21116': 3.032e-7, '13295': 1.7689e-7, '16846': 5.608e-7, '16115':  1.10127e-7}
    beta_mapping = {}
    for key, coefficients in coefficients_mapping.items():
        beta_mapping[key] = {k: (v * lambda_xray_mapping[key]) / (4 * np.pi)*1e6 for k, v in coefficients.items()}
    # print(beta_mapping)
    # required_dataset=['16846'] #,'13295','16846','16115','19505']
    dataset_list.sort(key=sort_dataset) 
    dataset='21116'
    axis = "x"
    for slices in dataset_list:
        for scale in delta_scale:
            
            # if '3_10_2_3_0_25' not in slices:
            #     continue
            phase_shift = True
            
            pixel_size = 0.3e-6  # Pixel size in meters (x, y)
            delta_z = pixel_size  # Voxel size in meters (z-direction)
            z = 4.9e-3  # distance from the sample to the detector
            betas = beta_mapping[dataset].copy()
            betas[2]=loop_mapping[dataset]
            betas[1]=betas[1]*0.95
            betas[3]=betas[3]*0.95
            
            refractive_indices = refractive_indices_mapping[dataset].copy()
            for key in refractive_indices:
                refractive_indices[key]=refractive_indices[key] * scale
            coefficients = coefficients_mapping[dataset]
            print('refractive_indices:',refractive_indices)

            print('betas:',betas)
            
            background =background_mapping[dataset]
            background_dir=os.path.join(background_dirs,str(background))
            lambda_xray = lambda_xray_mapping[dataset]
            slices=os.path.join(slices_dir,slices)

            model2, t_loading = model_loading(slices, dataset)
            
            background_img_list = background_extraction(background_dir)
            background_list =[os.path.join(background_dir, img) for img in background_img_list]
            save_dir = os.path.join(proj_dir,os.path.basename(slices)).replace('slices','projection')
            
            basename =os.path.basename(save_dir)
            rootname = os.path.dirname(save_dir)
            save_dir =os.path.join(rootname,str(f'delta_scale_{scale}'),basename)
            log_save_dir = os.path.join(rootname,str(f'delta_scale_{scale}'))
            print('save_dir:',save_dir)
            os.makedirs(save_dir,exist_ok=True)

            angle_list=np.linspace(0,180,len(background_list))
            t1 = time.time()
        
            parallel_simulation(angle_list,  model2, refractive_indices, betas, lambda_xray, pixel_size, delta_z, z, background_list, axis,save_dir,num_processes=32)
            t2 = time.time()
            log_metadata(log_save_dir, special_note, slices_dir, background_dir, pixel_size, betas, refractive_indices, slices, t1, t2)
            print("=====================================================")
            print(f"Time for {os.path.basename(save_dir)} is {t2 - t1} seconds")
            print("=====================================================")
        