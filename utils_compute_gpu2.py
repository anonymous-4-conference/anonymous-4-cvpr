import os
import numpy as np
import cupy as cp
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from cupy.fft import fft2, ifft2
import time
import pdb
from utils_projection import model_loading, compare_images, rotate_model
from utils_projection import sort_dataset, background_extraction, crop_to_fit, log_metadata, get_delta_beta, energy2wavelength
import os
import matplotlib.pyplot as plt
# Utility Functions
def check_gpu_memory(array_size):
    free_mem, total_mem = cuda.mem_get_info()
    required_mem = array_size * np.float32().nbytes
    return free_mem > required_mem


def simulate_xray_propagation_worker_pycuda(angle, sample, delta, beta, lam, pixel_size, delta_z, z, background_img, save_pth, axis):
    # Rotate model
    sample = rotate_model(sample, -angle)
    
    # Allocate PyCUDA arrays
    sample_gpu = gpuarray.to_gpu(sample.astype(np.float32))
    size_y, size_x = sample.shape[-2:]
    wavefield_real = gpuarray.to_gpu(np.ones((size_y, size_x), dtype=np.float32))
    wavefield_imag = gpuarray.zeros_like(wavefield_real)

    # Load and process background
    backg = (cv2.imread(background_img, 2)).astype(np.float32) / 65535
    background = np.sqrt(backg)
    background_gpu = gpuarray.to_gpu(background.astype(np.float32))

    # Simplified PyCUDA kernel for wave propagation
    kernel_code = """
    __global__ void propagate_wavefield_kernel(float *wavefield_real, float *wavefield_imag, float *background, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            wavefield_real[idx] *= background[idx];
            wavefield_imag[idx] *= background[idx];
        }
    }
    """
    mod = SourceModule(kernel_code)
    propagate_wavefield_kernel = mod.get_function("propagate_wavefield_kernel")

    # Kernel execution
    propagate_wavefield_kernel(
        wavefield_real, wavefield_imag, background_gpu,
        np.int32(wavefield_real.size),
        block=(256, 1, 1), grid=(wavefield_real.size // 256 + 1, 1, 1)
    )

    wavefield_final = wavefield_real.get()**2 + wavefield_imag.get()**2
    
    cv2.imwrite(save_pth, wavefield_final.astype('float32'))
    return wavefield_final


def compute_propagator(coarse_projection, delta_z, lambda_xray, pixel_size, fresnel=True):
    M, N = coarse_projection.shape
    k = 2 * np.pi / lambda_xray

    fx = cp.fft.fftfreq(N, d=pixel_size)
    fy = cp.fft.fftfreq(M, d=pixel_size)
    FX, FY = cp.meshgrid(fx, fy)

    if fresnel:
        result = cp.exp(1j * 2 * np.pi / lambda_xray * delta_z) * \
            cp.exp(-1j * np.pi * lambda_xray * delta_z * (FX**2 + FY**2))
    else:
        result = cp.exp(1j * 2 * np.pi / lambda_xray * delta_z) * cp.exp(1j * 2 * np.pi /
                                                                         lambda_xray * delta_z * cp.sqrt(1 - (FX * lambda_xray) ** 2 - (FY * lambda_xray) ** 2))

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

        beta_slice[sample_slice == value] = coefficients_mapping.get(
            value, -1) * pixel_size * 1e6
    T = cp.exp(-beta_slice)
    return T


def propagate_wavefield(u, propagator):
    u_ft = fft2(u)
    u_ft *= propagator
    u = ifft2(u_ft)
    return u


def simulate_xray_propagation_cupy(angle, sample, delta, beta, lam, pixel_size, delta_z, z, background_img, save_pth, axis,):
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
    backg = crop_to_fit(backg, wavefield)
    background = cp.sqrt(cp.array(backg))
    wavefield_final = wavefield.copy()
    wavefield_final = background * wavefield_final
    propagator = compute_propagator(
        coarse_projection, delta_z, lam, pixel_size)
    free_space_propagator = compute_propagator(
        coarse_projection, z, lam, pixel_size)
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
        T = compute_transmission_function(
            sample_slice, delta, beta, lam, pixel_size)
        wavefield *= T
        # wavefield = propagate_wavefield(wavefield, propagator)
        wavefield_final *= T
        wavefield_final = propagate_wavefield(wavefield_final, propagator)
        # print(cp.abs(wavefield_final)**2)

    wavefield_0 = wavefield.get()
    wavefield_0 = np.abs(wavefield_0)**2

    # wavefield_final_0 = wavefield_final.get()
    wavefield_final_0 = propagate_wavefield(
        wavefield_final, free_space_propagator).get()
    wavefield_final_0 = np.abs(wavefield_final_0)**2

    cv2.imwrite(save_pth, (wavefield_final_0.astype('float32')))

    return wavefield_0


def simulate_xray_propagation(angle_list, model, refractive_indices, betas, lambda_xray, pixel_size, delta_z, z, background_list, save_dir, axis):
    raw_proj = []
    sample_size = np.prod(model.shape)
    
    if check_gpu_memory(sample_size):
        print("Using PyCUDA for processing.")
        backend_worker = simulate_xray_propagation_worker_pycuda
    else:
        print("Using CuPy for processing.")
        backend_worker = simulate_xray_propagation_cupy
    
    for i, angle in enumerate(angle_list):
        save_pth = os.path.join(save_dir, f'projection_syris_{str(i).zfill(4)}.tiff')
        t1 = time.time()
        proj_img_cupy = simulate_xray_propagation_cupy(angle, model, refractive_indices, betas, lambda_xray, pixel_size, delta_z, z, background_list[i], save_pth, axis)
        t2 = time.time()
        proj_img_pycuda = simulate_xray_propagation_worker_pycuda(angle, model, refractive_indices, betas, lambda_xray, pixel_size, delta_z, z, background_list[i], save_pth, axis)
        t3 = time.time()
        print(f"CuPy: {t2 - t1} seconds, PyCUDA: {t3 - t2} seconds")


        proj_img = backend_worker(angle, model, refractive_indices, betas, lambda_xray, pixel_size, delta_z, z, background_list[i], save_pth, axis)
        raw_proj.append(proj_img)
    
    return raw_proj
