import os
import time
import numpy as np
from datetime import datetime
import re
import cv2  
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from skimage import measure
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import tqdm
import pdb

import periodictable
import periodictable.xsf
mapping_overall={
    "13076": {197: 0, 105: 1, 76: 3, 29: 2, 0: 0},
    "13304": {197: 0, 225: 1, 174: 2, 83: 3, 0: 0},
    "13284": {197: 0, 150: 1, 139: 2, 76: 3, 0: 0},
    "13668": {197: 0, 29: 1, 76: 2, 174: 3, 0: 0},
    "13313": {197: 0, 76: 2, 0: 0},
    "13295": {197: 0, 226: 1, 76: 2, 139: 3, 0: 0},
    "21116": {197: 0, 226: 1, 76: 2, 139: 3, 0: 0},
    "14116": {197: 0, 29: 1, 150: 2, 76: 3, 226: 4, 0: 0},
    "19505": {197: 0, 226: 1, 76: 2, 139: 3, 0: 0},
    "19485": {197: 0, 226: 1, 76: 2, 139: 3, 0: 0},
    "16846": {197: 0, 219: 1, 76: 2, 142: 3, 0: 0},
    "19431": {197: 0, 226: 1, 76: 2, 105: 3, 0: 0},
    "19488": {197: 0, 76: 1, 139: 2, 0: 0},
    "19441": {197: 0, 76: 1, 139: 2, 0: 0},
    "19494": {197: 0, 76: 1, 139: 2, 0: 0},
    "16115": {197: 0, 139: 2, 0: 0},
    "16102": {197: 0, 139: 2, 0: 0},
    "16000": {197: 0, 219: 1, 76: 2, 174: 3, 0: 0},
    "19224": {197: 0, 76: 1, 226: 2, 142: 3, 0: 0},
    "19413": {197: 0, 226: 1, 76: 2, 139: 3, 0: 0},
    "20467": {197: 0, 226: 1, 76: 2, 205: 3},
    "19835": {197: 0, 226: 1, 76: 2, 139: 3, 0: 0},
    "20072": {197: 0, 226: 1, 76: 2, 27: 3, 0: 0}, 
    "20453": {197: 0, 226: 1, 76: 2, 184: 3, 0: 0},
    "20506": {197: 0, 226: 1, 76: 2, 139: 3, 0: 0},
    "20048": {0: 0, 3: 1, 2: 2, 1: 3},
}




PLANCK_CONSTANT = 6.62607015e-34  # J·s
SPEED_OF_LIGHT = 2.99792458e8 # m/s
ELECTRON_CHARGE =  1.602176634e-19   # J/eV

def energy2wavelength(energy_eV):
    """
    Convert X-ray energy from electron volts (eV) to wavelength in angstroms (Å).

    Parameters:
    - energy_eV (float): X-ray energy in electron volts (eV).

    Returns:
    - wavelength (float): Wavelength of the X-ray in angstroms (Å).
    """
    return (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (energy_eV * ELECTRON_CHARGE) *1e10

def get_delta_beta(material_formula, energy_eV, density_g_cm3=None):
    """
    Calculate the Delta (δ) and Beta (β) components of the refractive index for a given material
    at a specified X-ray energy and density.

    Parameters:
    - material_formula (str): Chemical formula of the material (e.g., 'SiO2').
    - energy_eV (float): X-ray energy in electron volts (eV).
    - density_g_cm3 (float, optional): Density of the material in grams per cubic centimeter (g/cm³).
      If not provided, the default density from the `periodictable` package will be used.

    Returns:
    - delta (float) (phase shift): The real part of the refractive index decrement.
    - beta (float) (absorption): The imaginary part of the refractive index decrement.
    """
  
    compound = periodictable.formula(material_formula)
    energy_keV = energy_eV / 1000.0

    #sld : (float, float) | 10-6Å-2 https://periodictable.readthedocs.io/en/latest/api/xsf.html
    #so we need to convert it to cm^-2 by multiplying 1e10
    sld_real, sld_imag = periodictable.xsf.xray_sld(compound, density=density_g_cm3, energy=energy_keV)
    
    # Convert scattering length density to cm^-2
    sld_real_cm = sld_real * 1e10
    sld_imag_cm = sld_imag * 1e10

    wavelength_cm = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (energy_eV * ELECTRON_CHARGE) *1e2
    delta = (sld_real_cm) * wavelength_cm**2 / (2 * np.pi)
    beta = (sld_imag_cm) * wavelength_cm**2 / (2 * np.pi)
    return  beta, delta


def model_loading(model_dir,dataset,label=True,real=True,afterfix='tif',inverse=False):

    img_list = [path for path in os.listdir(model_dir) if afterfix in path ]
    # for i,path in enumerate(img_list):
    #     if '_000' in path:
    #         img_list[i]=path.replace('_000','')
    def extract_number(filename):
        match =  re.findall(r'(\d+)', filename)
       
        return int(match[-1]) if match else None
    img_list.sort(key=extract_number)
    if inverse:
        img_list=img_list[::-1]
    image_paths = [os.path.join(model_dir, img) for img in img_list]
    # pdb.set_trace()
    z_shape =len(img_list)
    init= cv2.imread(os.path.join(model_dir, img_list[0]),2)
    y_shape, x_shape = init.shape
    # print('Debug: sorted list is  {}'.format([name for name in img_list[:6]] ))

    if real:
        target_shape = (max(y_shape, x_shape), max(y_shape, x_shape))
        if label:
            model = np.zeros((z_shape, max(y_shape, x_shape), max(y_shape, x_shape)), dtype=np.uint8)
        else:
            sample_image=load_image(image_paths[0])
            model = np.zeros((z_shape, max(y_shape, x_shape), max(y_shape, x_shape)), dtype=sample_image.dtype)
    else:
        target_shape = (y_shape, x_shape)
        model = np.zeros((z_shape, y_shape, x_shape), dtype=np.uint8)


        
       
    t1 = time.time()
    # model2 = np.zeros((z_shape, y_shape, x_shape), dtype=np.uint8)
    # for i, img in enumerate(img_list):

    #     model2[i, :, :] = cv2.imread(os.path.join(model_dir, img),2)
    # model2=value_convert(model2,mapping_overall[dataset])

    # pdb.set_trace()
    thread_target_shapes = [target_shape] * len(image_paths)

    with ProcessPoolExecutor(max_workers=int(os.cpu_count()/2)) as executor:
        futures = []
        for image_path, target_shape in zip(image_paths, thread_target_shapes):
            futures.append(executor.submit(load_image, image_path, target_shape))

        for i, future in enumerate(tqdm.tqdm(futures, desc="Loading images")):
            img = future.result()
            if img is not None:
                model[i, :, :] = img


    
    # # with ThreadPoolExecutor() as executor:
    # #     images = list(executor.map(load_image, image_paths,thread_target_shapes))
    # for i, future in enumerate(futures):
    #     img=future.result()
    #     if img is not None: 
    #         model[i, :, :] = img
    model_cp =model.copy()
    if label:
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax.imshow(model[413, :, :])
        # ax.set_title('Model')
        # plt.show()
        # pdb.set_trace()
        try:
            model2=value_convert(model_cp,mapping_overall[dataset])
            assert np.array_equal(np.unique(model2), np.array([0, 1, 2, 3])) or np.array_equal(np.unique(model2), np.array([0, 2]))
            print('Loading label model: Model values are converted')

        except:
            print(f'{dataset} is not in the mapping list')
            plt.imshow(model[400,:,:])
            plt.show()
            pdb.set_trace()
            print(f'Error: {dataset} model values are not correct')
            print(f"swapping the values of crystal and the mother liquor")

            model2 = model.copy()
            model2[model2 == 1] = 30
            model2[model2 == 3] = 1
            model2[model2 == 30] = 3
            plt.imshow(model2[400,:,:])
            plt.show()
            pdb.set_trace()
    else:
        if model.max() <= 1:
            model=normalize_image(model).astype(np.uint8)
        model2=model
        print('Loading training model: Model values are not converted')
    t2 = time.time()
    t_loading = t2 - t1
    # print('model loading time:', t2 - t1)
    # pdb.set_trace()
    return model2,t_loading

def robust_scale(image):
    median = np.median(image)
    iqr = np.percentile(image, 75) - np.percentile(image, 25)
    return (image - median) / iqr

def z_standardize(image):
    return (image - np.mean(image)) / np.std(image)

def normalize_image(image, new_min=0, new_max=255):
    """
    Normalize the intensity values of a grayscale image to span from new_min to new_max.

    Parameters:
    - image: NumPy array representing the grayscale image to be normalized.
    - new_min: The minimum value of the normalized intensity range. Default is 0.
    - new_max: The maximum value of the normalized intensity range. Default is 255.

    Returns:
    - normalized_image: NumPy array representing the normalized image.
    """
    # Find the current minimum and maximum values of the image
    current_min = np.min(image)
    current_max = np.max(image)
    
    # Normalize the image to the range [new_min, new_max]
    normalized_image = (image - current_min) / (current_max - current_min) * (new_max - new_min) + new_min
    
    return normalized_image

def crop_to_fit(background_img, projected_img):
    center_y, center_x = background_img.shape[0] // 2, background_img.shape[1] // 2
    height, width = projected_img.shape

    y_start = max(center_y - height // 2, 0)
    x_start = max(center_x - width // 2, 0)

    cropped_background = background_img[y_start:y_start + height, x_start:x_start + width]
    return cropped_background

def compare_images(syn_img, real_img):
    # Read images

    # Compute SSIM
    ssim_score,gradient,full_image = ssim(syn_img, real_img,gradient=True, full=True)
    # Compute MSE
    mse_score = mse(syn_img, real_img)

    # Compute PSNR
    psnr_score = psnr(syn_img, real_img)

    return ssim_score, mse_score, psnr_score

def load_image(path,target_shape=None):

    img= cv2.imread(path, 2)
    if target_shape is not None:
        if img.shape[0] != target_shape[0] or img.shape[1] != target_shape[1]:
            x_shape, y_shape = img.shape
            target_shape = (max(x_shape, y_shape), max(x_shape, y_shape))
            img = padding_image(img, target_shape)

            return img
        else:
            return img
    else:
        return img
def preprocess_model(model):
    dim_z, dim_y, dim_x = model.shape
    side=max(dim_y,dim_x)
    new_model = np.zeros((dim_z,side,side))

    for i in range(dim_z):
        pad_y = (side - dim_y) // 2
        pad_x = (side - dim_x) // 2
        if (side - dim_y) % 2 == 0:
            pad_width_y = (pad_y, pad_y)
        else:
            pad_width_y = (pad_y, pad_y + 1)
        
        if (side - dim_x) % 2 == 0:
            pad_width_x = (pad_x, pad_x)
        else:
            pad_width_x = (pad_x, pad_x + 1)
        slice_padded = np.pad(model[i, :, :], (pad_width_y, pad_width_x), mode='constant', constant_values=0).astype(np.uint8)
        new_model[i,:,:] = slice_padded
    return new_model.astype(np.uint8)

def convert_to_syris_mesh_format(model):
    verts, faces, _, _ = measure.marching_cubes(model, level=0)
    # Preallocate an array for the triangle vertices. Each face consists of 3 vertices,
    # and there are 3 coordinates (x, y, z) for each vertex.
    # The shape is (3, faces.shape[0] * 3) because each face is a triangle.
    # pdb.set_trace()
    triangles = np.zeros((3, faces.shape[0] * 3), dtype=verts.dtype)
    
    for i, face in enumerate(faces):
        # For each face, retrieve the vertices that form the triangle
        for j in range(3): # Each face has 3 vertices
            triangles[:, i * 3 + j] = verts[face[j], :]
    
    return triangles

def background_extraction(background_dir,threshold=30):
    def extract_number(filename):
        match =  re.findall(r'(\d+)', filename)
       
        return int(match[-1]) if match else None
    image_list=[]
    for filename in os.listdir(background_dir):
        
        index = extract_number(filename)
        
        
        if index < threshold:
            continue
        
        image_list.append(filename)
       
    sorted_image_list = sorted(image_list, key=extract_number)
    
    return sorted_image_list

def rotate_image ( image , angle,image_center=None ) :
    if image_center is None:
        image_center = tuple( np.array( image.shape[1 : :-1] ) / 2 )
    # pdb.set_trace()
    rot_mat = cv2.getRotationMatrix2D( image_center , angle , 1.0 )

    result = cv2.warpAffine(
        image , rot_mat , (image.shape[1] , image.shape[0]) , flags = cv2.INTER_NEAREST )
    return result

def rotate_model ( model , angle,center=None) :
    new_model = np.zeros_like(model)
    for i in range(model.shape[0]):
        # pdb.set_trace()
        new_model[i,:,:] = rotate_image(model[i,:,:], angle,image_center=center )

    return new_model

def padding_image(image, target_shape):
    target_y, target_x = target_shape

    y, x = image.shape
    pad_y = max(target_y - y, 0)
    pad_x = max(target_x - x, 0)

    # Divide the padding to pad equally on both sides
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left 
    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    return padded_image


def value_convert( array,mapping):
  
    if mapping is not None :
        for i, value in enumerate(mapping):
            array[array == value] = mapping[value]
            # p
        return array
    else:
        unique_values = np.unique(array)
        sorted_indices = np.argsort(-unique_values) 
        value_to_sequence = {value: i for i, value in enumerate(unique_values[sorted_indices])}

        sequence_array = np.vectorize(value_to_sequence.get)(array)
        return sequence_array

def list_only_directories(parent_directory):
    directories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    return directories

def sort_dataset(name):
    try:
        parts = re.findall(r'\d+', name)
     
        return (int(parts[1]),int(parts[2]),int(parts[-1]),int(parts[0]))
    except:
        return name

# Define your logging function
def log_metadata(proj_dir, special_note, slices_dir, background_dir, pixel_size, coefficients, refractive_indices, slices, start_time, end_time):
    # Create metadata dictionary
    metadata = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'special_note': special_note,
        'slices_dir': slices_dir,
        'background_dir': background_dir,
        'proj_dir': proj_dir,
        'pixel_size': pixel_size,
        'coefficients': coefficients,
        'refractive_indices': refractive_indices,
        'slices_processed': slices,
        'processing_time_seconds': end_time - start_time
    }
    
    # Define file path for logging
    log_file = os.path.join(proj_dir, 'metadata_log.txt')
    
    # Write metadata to file
    with open(log_file, 'w+') as f:
        f.write("Metadata for project:\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

