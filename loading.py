import h5py
import pdb
import numpy as np
import matplotlib.pyplot as plt
dataset = 16846
for dataset in [16846,21116,13295]:
    image_path =f'/home/yishun/simulation/test/paper3/deformableLKA/3D/tomopy/test_{dataset}_real/{dataset}_real_pred.h5'
    h5f = h5py.File(image_path, 'r') #+"/mri_norm2.h5", 'r')
    train = h5f['train'][:]
    label = h5f['label'][:]
    np.save(f'/home/yishun/simulation/test/paper3/deformableLKA/3D/tomopy/{dataset}_real_pred_train.npy', train)
pdb.set_trace()