import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import re
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pdb
import monai.transforms as mt   
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate,
    RandScaleIntensity,
    RandShiftIntensity,
    RandGaussianNoise,
    ToTensor,
    CastToType,
)
from monai.transforms import Resize as MonaiResize




def process_file(train_name, file_name, resize_size, interpolation):

    # Read data from HDF5 file
    with h5py.File(train_name, 'r') as h5f:
        train = h5f['train'][:]
        label = h5f['label'][:]
    origin_shape=train.shape
    # Resize the data
    # train = Resize((resize_size, resize_size, resize_size)).resize_image(train, interpolation) 
    # label = Resize((resize_size, resize_size, resize_size)).resize_image(label, 0)
    train = cascade_downscale(train, resize_size, interpolation)
    label = cascade_downscale(label, resize_size, 0)  
    # Write the processed data to a new HDF5 file
    with h5py.File(file_name, 'w') as h5_file:
        h5_file.create_dataset('train', data=train)
        h5_file.create_dataset('label', data=label)
    
    return file_name

def cascade_downscale(data: np.ndarray, target_size: int, interpolation: int) -> np.ndarray:
    current_shape = data.shape
    steps = []

    # Determine scaling steps
    if all(dim >= target_size * 4 for dim in current_shape):
        steps = [dim // 2 for dim in current_shape]  # Half size
        steps.append(target_size)  # Final size
    elif all(dim >= target_size * 2 for dim in current_shape):
        steps = [target_size]  # Directly to final size if no room for 1/4 scaling

    for step in steps:
        data = Resize((step, step, step)).resize_image(data, interpolation)

    return data


class Tomodataset(Dataset):
    def __init__(self, base_dir=None,test_dataset=['21116','19835','20048','21944'],only_target=None, common_transform=None,sp_transform=None,syn_record_file='record_syn.xlsx',real_record_file='record_real.xlsx',train=False,train_classes={'crystal':3, 'background':0, 'loop':2, 'liquor':1},interpolation=0,args=None,pre_interpolate=True,test=False):
        if args is not None:
            self.args = args
        self.test = test
        self._base_dir = base_dir
        print("Loading images from Base dir: {}".format(self._base_dir))
        self.common_transform = common_transform
        self.sp_transform = sp_transform
        self.sample_list = []
        self.syn_record_file = os.path.join(base_dir,syn_record_file)
        self.real_record_file =  os.path.join(base_dir,real_record_file)
        self.train = train  
        self.test_dataset = test_dataset
        self.syn_train_list,self.real_train_list,self.test_list = self._load_train_paths() 
        self.syn_train_list = self.replace_base_dir(self.syn_train_list)
        self.real_train_list = self.replace_base_dir(self.real_train_list)
        self.test_list = self.replace_base_dir(self.test_list)
        self.only_target = only_target
        self.pre_interpolate = pre_interpolate
            
        self.train_classes = train_classes
        self.num_classes = len(self.train_classes)
        if train:
            self.train_list = self.real_train_list + self.syn_train_list 
            print("Number of training samples: {}".format(len(self.train_list)))
        else:
            self.train_list = self.test_list
            print("Number of testing samples: {}".format(len(self.train_list)))
        if self.only_target is not None:
            self.train_list = [self.only_target]
            print("Number of target samples: {}".format(len(self.train_list)))

        #preprocess data for faster training
        self.interpolation = interpolation
        if pre_interpolate:
        # if self.interpolation == 0:
        #     pass
        # else:
            self.train_list  = self.preprocessh5(self.train_list, self.interpolation)
        
    def preprocessh5(self,train_list,interpolation,save_dir=None):
        new_train_list = []
        if save_dir is None:
            save_dir = os.path.join(self._base_dir,'preprocessed')
        else:
            save_dir = save_dir
        # shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir,exist_ok=True)
        # ===testing of single process================
        # for train_name in train_list:
        #     print(f"Processed file: {train_name}")
        #     dataset_name = os.path.basename(train_name)
        #     inputfile_name = os.path.join(save_dir, dataset_name)
        #     if os.path.exists(inputfile_name):
        #         new_train_list.append(inputfile_name)
        #         continue
        #     try:
        #         file_name=process_file(train_name, inputfile_name, self.args.resize_size, interpolation)
        #     except:
        #         pdb.set_trace()
        #     new_train_list.append(file_name)
        # ===testing of single process================   
        # pdb.set_trace()
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            for train_name in tqdm.tqdm(train_list, desc=f"Counting Train:{self.train} data"):
                dataset_name = os.path.basename(train_name)
                inputfile_name = os.path.join(save_dir, dataset_name)
                if os.path.exists(inputfile_name):
                    new_train_list.append(inputfile_name)
                    continue
                futures.append(executor.submit(process_file, train_name, inputfile_name, self.args.resize_size, interpolation))
            
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Preprocessing files"):
                result = future.result()
                print(f"Processed file: {result}")
                new_train_list.append(result)
        
        return new_train_list
    
    def replace_base_dir(self,img_list):
        new_list = []
        for i in range(len(img_list)):
            data_pth = img_list[i]
            data =os.path.basename(data_pth)
            new_pth = os.path.join(self._base_dir,data)
            if os.path.exists(new_pth):
                new_list.append(new_pth)
        return new_list

    def _load_train_paths(self):
        only_syn=False
        only_real=False
        syn_train_list = []
        real_dataset={}
        real_train_list =[]
        test_list = []
        # Load synthetic train paths
        if self.syn_record_file and os.path.exists(self.syn_record_file):
            syn_df = pd.read_excel(self.syn_record_file).dropna() 
            syn_paths = syn_df['New Path'].tolist()
            syn_train_list.extend(syn_paths)
        
        
        # Load real train paths
        if self.real_record_file and os.path.exists(self.real_record_file):
            real_df = pd.read_excel(self.real_record_file).dropna() 
            
            for row in real_df.iterrows():
                raw_data=row[1]['Original Path']
                dataset=re.findall(r'\d+',os.path.basename(raw_data))[0]
                
                scales = re.findall(r'\d+\.\d+|\d+', os.path.basename(os.path.dirname(os.path.dirname(raw_data))))
                if len(scales) == 0:
                    scale='real'
                else:
                    scale = scales[0]
                if scale not in real_dataset:
                    real_dataset[scale] = {}
                if dataset not in real_dataset[scale]:
                    real_dataset[scale][dataset] = {}
                
                real_dataset[str(scale)][str(dataset)]['raw']=raw_data
                real_dataset[scale][dataset]['new']=row[1]['New Path']
        

        for scale in real_dataset.keys():
            
            for dataset in real_dataset[scale].keys():
                # if  scale == '0.01': #scale == '0.1' or scale == '0.5' or

                    
                if scale =='1' or scale == '0.5': # or scale == '0.1' or scale == '0.01':
                    continue    
                # if dataset in self.test_dataset:
                #     test_list.append(real_dataset[scale][dataset]['new'])
                #     continue
                
                real_train_list.append(real_dataset[scale][dataset]['new'])
        # pdb.set_trace() 
        if only_real:
            syn_train_list=[]
            real_train_list=[]
        for file in os.listdir(self._base_dir):
            if '.h5' in file and "Image" not in file and 'sirt' not in file:
                    pth=os.path.join(self._base_dir,file)

                    if only_syn is False:
                        real_train_list.append(pth)
                    for i in self.test_dataset:
                        if i in file:
                            # pdb.set_trace()
                            if self.test:
                                if 'real' in file and  'pred' in file:
                                    test_list.append(pth)
                            else:
                                test_list.append(pth)
                                if only_syn is False:
                                    real_train_list.remove(pth)

            # if 'fbp' in file:
            #     pth=os.path.join(self._base_dir,file)
            #     real_train_list.append(pth)
                # syn_train_list=[]


        # pdb.set_trace()
        # real_train_list=[]
        return syn_train_list,real_train_list,test_list

    def __len__(self):
        return len(self.train_list) 
    
    def print_sample(self,logger):
        logger.info("self.train_list: {}".format(self.train_list))
    

            
    def __getitem__(self, idx):
        #print("Index: {}".format(idx))
        train_name = self.train_list[idx]
        dataset_name = os.path.basename(train_name)
        h5f = h5py.File(train_name, 'r') #+"/mri_norm2.h5", 'r')
        train = h5f['train'][:]
        label = h5f['label'][:]
        assert train.shape == label.shape
        origin_shape=train.shape
        # if 'syn' in dataset_name:
        #     train=np.transpose(train,(2,1,0))
        #     label=np.transpose(label,(2,1,0))
        # else:
        #     train=np.transpose(train,(1,2,0))
        #     label=np.transpose(label,(1,2,0))
        # if 'real' in train_name:
        #     one_hot_label = self.binary_mask(label, real=True)
        # else:
        #     one_hot_label = self.binary_mask(label, real=False)
        # before_sample = {'train': train, 'label': label}
        # if self.common_transform:
        #     before_sample = self.common_transform(before_sample)
        # if self.common_transform is not None:
        transformed_label = self.common_transform(label)
        transformed_train = self.common_transform(train)
        transformed_train =torch.unsqueeze(transformed_train,0)
        transformed_label=torch.unsqueeze(transformed_label,0)
        
        if self.pre_interpolate is False:
            transformed_train = MonaiResize((self.args.resize_size,self.args.resize_size,self.args.resize_size),mode='bilinear')(transformed_train)
            transformed_label =MonaiResize((self.args.resize_size,self.args.resize_size,self.args.resize_size),mode="nearest")(transformed_label)

        # pdb.set_trace()
        if self.train:
            random_x = np.random.uniform(0,2*np.pi) #Range of rotation angle in radians
            # random_x=0
            
            random_y = np.random.uniform(0,np.pi)
            random_y =0 # in crystallography, we only rotate around x-axis
            
            random_z = np.random.uniform(-np.pi/6,np.pi/6) #in tomography of crystallography, we may rotate around z-axis for a little bit
            # random_z=0
            # random_z = np.random.uniform(0,180)
            # print(f"random_x: {random_x*180/np.pi}, random_y: {random_y*180/np.pi}") 
            
            rotate_func=RandRotate(
                    range_x=(random_x, random_x), 
                    range_y=(random_y, random_y),
                    range_z=(random_z, random_z),
                    prob=1,
                    keep_size=True,
                    mode='bilinear'
                )
            transformed_train= rotate_func(transformed_train)
            # Randomly scale intensity for brightness variation
            transformed_train = RandScaleIntensity(factors=0.1, prob=0.5)(transformed_train)
            # # Randomly shift intensity for contrast variation
            transformed_train = RandShiftIntensity(offsets=0.1, prob=0.5)(transformed_train)
            # # Add Gaussian noise for robustness to noise
            transformed_train = RandGaussianNoise(mean=0.0, std=0.05, prob=0.5)(transformed_train)
            # transformed_train=torch.squeeze(transformed_train,0)
            
            
            label_rotate_func=RandRotate(
                    range_x=(random_x, random_x), 
                    range_y=(random_y, random_y),
                    range_z=(random_z, random_z),
                    prob=1,
                    keep_size=True,
                    mode='nearest'
                )
            transformed_label= label_rotate_func(transformed_label)
            
        transformed_label=torch.squeeze(transformed_label,0)
        
        # pdb.set_trace()
        # transformed_train =torch.unsqueeze(transformed_train,0)
        # sample = {'train': transformed_train, 'label': transformed_label,} #,'onepdb
        # hot_label':one_hot_label

        # if self.common_transform:
        #     sample = self.common_transform(sample)

        # if 'real' in train_name:
            # one_hot_label = self.binary_mask(sample['label'], real=True)
        # else:
        # one_hot_label = self.binary_mask(sample['label'], real=False)
        one_hot_label = self.binary_mask(transformed_label, real=False)
        
        sample = {'train': transformed_train, 'label': transformed_label,'onehot_label':one_hot_label}
        # if self.sp_transform:
            # sample = self.sp_transform(sample)
        
        
        # pdb.set_trace()
        if self.test:
            onehot_predict = self.binary_mask(sample['train'], real=False)
            sample['label'] = onehot_predict

        sample['image_path'] = train_name
        sample['original_shape'] = origin_shape
        # pdb.set_trace()
        return  sample

    def binary_mask(self, label, real=True):
        """
        Create binary masks for each class excluding the background (pixel value 0).
        Args:
        - label_image (np.ndarray): The label image where pixel values represent different classes.

        Returns:
        - binary_masks (np.ndarray): A binary mask for each class excluding the background.
        """
        # Define class pixel values excluding the background (0)
        # class_values = np.unique(label)[1:]
        binary_masks = []
        if self.num_classes==4:
            if real:
                class_values = [0,85, 170, 255]
            else:
                class_values = [0,1,2,3]
            # Initialize an empty list to store binary masks
            for i in range(len(class_values)):
                
                # Create a binary mask for the current class
                binary_mask = ((label == class_values[i])).astype(np.uint8)
                binary_masks.append(binary_mask)
        else:
            assert self.num_classes<4
            counter_binary_mask = ((label >0)).astype(np.uint8)
            binary_mask =1- counter_binary_mask
            binary_masks=[binary_mask]
            for i,key in enumerate(self.train_classes.keys()):
                label_value = self.train_classes[key]
                if key == 'background':
                    continue
                else:
                    binary_mask = ((label ==label_value)).astype(np.uint8)
                binary_masks.append(binary_mask)
        

        
        # Stack the binary masks along a new axis
        binary_masks = np.stack(binary_masks, axis=0)
        
        return binary_masks




class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        train, label = sample['train'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            train = np.pad(train, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = train.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        train = train[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'train': train, 'label': label}


class Resize(object):
    """
    Resize the train and label in a sample to a given size using nearest neighbor interpolation.
    Args:
    output_size (int or tuple): Desired output size
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)), "Output size should be an int or tuple"
        self.output_size = output_size

    def resize_image(self, image,order=0):

        new_h, new_w, new_d =self.output_size
        # pdb.set_trace()
        zoom_factors_train = (new_h / image.shape[0], new_w / image.shape[1], new_d / image.shape[2])

        # Resize train and label using nearest neighbor interpolation
        train_resized = zoom(image, zoom_factors_train, order=order)
        return train_resized
    def __call__(self, sample):
        train, label = sample['train'], sample['label']
        if self.output_size == train.shape:
            return {'train': train, 'label': label}

        new_h, new_w, new_d = self.output_size
        
        # Calculate the zoom factors for each dimension
        zoom_factors_train = (new_h / train.shape[0], new_w / train.shape[1], new_d / train.shape[2])
        zoom_factors_label = (new_h / label.shape[0], new_w / label.shape[1], new_d / label.shape[2])

        # Resize train and label using nearest neighbor interpolation
        train_resized = zoom(train, zoom_factors_train, order=0) #
        label_resized = zoom(label, zoom_factors_label, order=0)
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(train_resized[256, :, :], cmap='gray')
        # ax[0].set_title('Resized Train')
        # ax[1].imshow(label_resized[256, :, :], cmap='gray')
        # ax[1].set_title('Resized Label')
        # plt.show()
        return {'train': train_resized, 'label': label_resized}
    
class RandomCrop(object):
    """
    Crop randomly the train in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        train, label = sample['train'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            train = np.pad(train, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = train.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        train = train[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'train': train, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        train, label = sample['train'], sample['label']
        k = np.random.randint(0, 4)
        train = np.rot90(train, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        train = np.flip(train, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'train': train, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        train, label = sample['train'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(train.shape[0], train.shape[1], train.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        train = train + noise
        return {'train': train, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        train, label = sample['train'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'train': train, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        train = sample['train']
        
        train = train.reshape(1, train.shape[0], train.shape[1], train.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'train': torch.from_numpy(train), 'label': torch.from_numpy(sample['label']),
                    'onehot_label': torch.from_numpy(sample['onehot_label'])}
        else:
            return {'train': torch.from_numpy(train), 'label': torch.from_numpy(sample['label'])}



