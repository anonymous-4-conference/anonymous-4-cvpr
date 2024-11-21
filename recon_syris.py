import numpy as np
import tomopy
import glob
import pdb
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import disk
# import open3d as o3d
import os
import cv2
import re
import time
from concurrent.futures import ThreadPoolExecutor
import h5py
import argparse 
from utils_projection import sort_dataset,list_only_directories,model_loading,padding_image
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



def flat_field_correction(projection_dir,flat_field_save_dir, background_dir,flat_field_name, darks,flat_field_name_needed,edge_enhance=False,save_img=False):
    def extract_number(filename):
        match =  re.findall(r'(\d+)', filename)
       
        return int(match[-1]) if match else None
    dark_fields = []
    projection_images=[]
    for f in os.listdir(projection_dir):
    
        if  'tif' in f:
            if 'syris' in  f:
                if edge_enhance:
                    if 'test' in f:
                        projection_images.append(f)
                else:
                    if 'test' not in f:
                        projection_images.append(f)
    # projection_images = [f for f in os.listdir(projection_dir) if f.endswith('.png')] 

    projection_images.sort(key=extract_number)
    
    sample_img = cv2.imread(os.path.join(projection_dir, projection_images[0]), cv2.IMREAD_UNCHANGED)
    # pdb.set_trace()
    # Load and process each dark field image
    flat_field=crop_to_fit(cv2.imread(os.path.join(background_dir,flat_field_name), cv2.IMREAD_UNCHANGED), sample_img)/65535
    for image_path in darks:
        # Load the image
        
        img = cv2.imread(os.path.join(background_dir,image_path), 2)
        # Crop the first 50 columns
        
        cropped_img =crop_to_fit(img, sample_img)
        # Append to the list
        dark_fields.append(cropped_img)
    dark_fields_stack = np.array(dark_fields)
    avg_dark_field = np.mean(dark_fields_stack, axis=0)/65535
    
    epsilon = 1e-10

    projection_images_list=[]
   
    for i,image_path in enumerate(projection_images):
        flat_field_name=flat_field_name_needed[i]
        img = cv2.imread(os.path.join(projection_dir,image_path), cv2.IMREAD_UNCHANGED)
 
        # flat_field=crop_to_fit(cv2.imread(os.path.join(background_dir,flat_field_name), cv2.IMREAD_UNCHANGED), sample_img)
        corrected_img = (img - avg_dark_field) / (flat_field - avg_dark_field + epsilon)
        if corrected_img.min() < 0:
            corrected_img = (img ) / (flat_field  + epsilon)
        # Save the corrected image
        corrected_img[corrected_img>1]=1
        if save_img:
            # corrected_img_normalized = (corrected_img - np.min(corrected_img)) / (np.max(corrected_img) - np.min(corrected_img))
            corrected_img_uint16 = np.uint16(corrected_img * 65535)
            # pdb.set_trace()
            # if i==150:
            #     pdb.set_trace()
            cv2.imwrite(os.path.join(flat_field_save_dir, f"corrected_{image_path}"), corrected_img_uint16)
        # tifffile.imwrite(os.path.join(flat_field_save_dir, f"corrected_{image_path.replace('.png','.tiff')}"), corrected_img)
        # imageio.imwrite(os.path.join(flat_field_save_dir, f"corrected_{image_path.replace('.png','.tiff')}"), corrected_img, format='TIFF')
        # print(f"corrected {image_path}")
        projection_images_list.append(corrected_img)
    #pdb.set_trace()
    return np.array(projection_images_list)

def background_extraction(background_dir,threshold=30):
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else None
    image_list=[]
    for filename in os.listdir(background_dir):
        
        index = extract_number(filename)
        
        
        if index < threshold:
            continue
        
        image_list.append(filename)
       
    sorted_image_list = sorted(image_list, key=extract_number)
    
   
    return sorted_image_list

def crop_to_fit(background_img, projected_img):
    center_y, center_x = background_img.shape[0] // 2, background_img.shape[1] // 2
    height, width = projected_img.shape

    y_start = max(center_y - height // 2, 0)
    x_start = max(center_x - width // 2, 0)

    cropped_background = background_img[y_start:y_start + height, x_start:x_start + width]
    return cropped_background

def reconstruction(img_list,options,file_name,center=None,slicemodel=None,syn=True):
    # pdb.set_trace()
    # img_list =-np.log(img_list)
    # pdb.set_trace()
    img_list = tomopy.minus_log(img_list)
    theta = np.linspace(0, np.pi, img_list.shape[0])
    # pdb.set_trace()
    # rot_center = tomopy.find_center_pc(img_list[0],img_list[-1])
    if center is None:
        rot_center = img_list.shape[-1]//2
    else:
        rot_center=center
    recon = tomopy.recon(img_list,theta,center=rot_center,algorithm=tomopy.astra,options=options,)
    recon_0 = tomopy.recon(img_list,theta,center=rot_center,algorithm=tomopy.astra,options=options,)
    # recon=recon[::-1,:,:]
    # recon =tomopy.remove_ring(recon)
    recon =tomopy.circ_mask(recon, axis=0, ratio=0.95)
    # recon =cv2.normalize(recon, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    recon =cv2.normalize(recon, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    for i in range(recon.shape[0]):
        # img_normalized_uint16 = cv2.normalize(recon[i], None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
        # img_normalized_uint8 = cv2.normalize(recon[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_normalized_uint8=recon[i]
        cv2.imwrite(os.path.join(os.path.dirname(file_name),f"recon_{str(i).zfill(4)}.png"), img_normalized_uint8)
    
    if slicemodel is not None:
        new_label =np.zeros_like(recon)
        for i in range(len(slicemodel)):
            slice_img=padding_image(slicemodel[i],recon[0].shape)
            if syn:
                slice_img=cv2.rotate(slice_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_label[i] =slice_img
    assert recon.shape==new_label.shape
    
    # fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    # ax[0].imshow(recon[100], cmap='gray')
    # ax[0].set_title('Reconstruction')
    # ax[1].imshow(new_label[100], cmap='gray')
    # ax[1].set_title('Label')
    # ax[2].imshow(recon[100]-new_label[100])
    # ax[2].set_title('Difference')
    # plt.show()
    # pdb.set_trace()
    with h5py.File(file_name, 'w') as h5_file:
        h5_file.create_dataset('train', data=recon)
        if new_label is not None:
            h5_file.create_dataset('label', data=new_label)
    
    del recon, img_list
if __name__ =="__main__":
    # proj_dir= '/mnt/data_smaller/yishun/simulation/projections/crystal_4_12_2_3_0_50'
    # background_dir='/home/yishun/simulation/test/background/20457'
    projection_dirs=args.proj_dir
    background_dir=args.background_dir
    slices_dirs='/mnt/data_smaller/yishun/simulation/4th_batch/slices'

    background_dirs='/mnt/data_smaller/yishun/simulation/background'
    projection_dirs='/mnt/data/yishun/simulation/7th_batch/projections_phase/'
    recon_dirs='/mnt/data/yishun/simulation/7th_batch/recon'
    projection_dirs='/mnt/data/yishun/simulation/8th_batch/projections_syn3/'
    recon_dirs='/mnt/data/yishun/simulation/8th_batch/recon_syn3'
    slices_list=[path for path in os.listdir(slices_dirs) if os.path.isdir(os.path.join(slices_dirs,path))] 
    slices_list.sort(key=sort_dataset) 
    projection_dirs_list= os.listdir(projection_dirs)
    # projection_dirs_list.sort(key=sort_dataset)
    background_mapping={'13295':20468,'21116':20518,'16846':20457}
    wavelength='21116'
    dataset=wavelength
    required_scale=[0.5,0.1,0.05,0.01]
    # for dataset in wavelength:
    for scale in projection_dirs_list:
        

        scale_proj_dirs=os.path.join(projection_dirs,scale)
        if not os.path.isdir(scale_proj_dirs):
            continue
        if float(scale.split('_')[-1]) not in required_scale:
            continue
        scale_proj_list = list_only_directories(scale_proj_dirs)
        scale_proj_list.sort(key=sort_dataset)

        
        for i,proj_dirs in enumerate(scale_proj_list):
            
            slices = slices_list[i]
            
            n=0
            while slices.replace('slices','projection')!=proj_dirs:
                try:
                    slices = slices_list[n]
                except:
                    pdb.set_trace()
                n+=1
            # os.path.join(slices_dirs, scale,os.path.basename(proj_dir)).replace('projection','slices')
            assert slices.replace('slices','projection')==proj_dirs
            

            proj_dir=os.path.join(scale_proj_dirs,proj_dirs)
            # proj_dir='/mnt/data_smaller/yishun/simulation/4th_batch/projections/projection_3_8_2_3_0_25'
            if not os.path.isdir(proj_dir):
                continue
            background_dir=os.path.join(background_dirs,str(background_mapping[dataset]))
            # recon_save_dir =os.path.join(os.path.dirname(os.path.dirname(proj_dir)),'recon')
            
            recon_save_dir=os.path.join(recon_dirs,  scale,os.path.basename(proj_dir)).replace('projection','recon')
            # pdb.set_trace()
            os.makedirs(recon_save_dir,exist_ok=True)
            h5_files = glob.glob(os.path.join(recon_save_dir,'*.h5'))
            if h5_files:
                print(f'{recon_save_dir} is done')
                continue
            else:

                print(f'\n {recon_save_dir} is not done and being processed ')
                print(f' label is {slices} ')
                print(f' projection is {proj_dir} \n')
                
            slices=os.path.join(slices_dirs,slices)

            model2, t_loading = model_loading(slices, dataset,real=False,label=False)
            # the padding is done in the later reconstruction function
            
            dataset_list=[path for path in os.listdir(proj_dir)]    
            dataset_list.sort()
            
            flat_field_name_all= background_extraction(background_dir, threshold=0)
            dark_field_names=flat_field_name_all[0:30]
            flat_field_name_needed=flat_field_name_all[30:]
            flat_field_name =flat_field_name_all[30]
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA','FilterType':'Triangular'}
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'} #'FilterType':'hamming'
            flat_field_save_dir = os.path.join(recon_save_dir,'flat_field')
            os.makedirs(flat_field_save_dir,exist_ok=True)
            
            file_name =os.path.join(recon_save_dir,f"recon_{os.path.basename(proj_dir)}.h5")
            t1 = time.time()
            img_list =flat_field_correction(proj_dir,flat_field_save_dir,background_dir, flat_field_name, dark_field_names,flat_field_name_needed,save_img=True,edge_enhance=False)
            # pdb.set_trace()
            # random_scaling_factors = np.random.uniform(-0.1, 0.1, img_list.shape)
            # scaled_img_list = img_list * (1 + random_scaling_factors)
            reconstruction(img_list,options,file_name,center=None,slicemodel=model2)
            t2 = time.time()
   
            print('recon time:', t2 - t1)   
            print(f'{file_name} is done')
            # pdb.set_trace()