import nnUnet
import os
import shutil
import gdown
mount_dir='/mnt/data/yishun/nnUnet'
base_dir='./'
os.makedirs(mount_dir, exist_ok=True)
def make_if_dont_exist(folder_path,overwrite=False):
    """
    creates a folder if it does not exists
    input: 
    folder_path : relative path of the folder which needs to be created
    over_write :(default: False) if True overwrite the existing folder 
    """
    if os.path.exists(folder_path):
        
        if not overwrite:
            print(f"{folder_path} exists.")
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)

    else:
      os.makedirs(folder_path)
      print(f"{folder_path} created!")

print("Current Working Directory {}".format(os.getcwd()))
path_dict = {
    "nnUNet_raw" : os.path.join(mount_dir, "nnUNet_raw"), 
    "nnUNet_preprocessed" : os.path.join(mount_dir, "nnUNet_preprocessed"), # 1 experiment: 1 epoch took 112s
    "nnUNet_results" : os.path.join(mount_dir, "nnUNet_results"),
    "RAW_DATA_PATH" : os.path.join(mount_dir, "RawData"), # This is used here only for convenience (not necessary for nnU-Net)!
}

# Write paths to environment variables
for env_var, path in path_dict.items():
  os.environ[env_var] = path 

# Check whether all environment variables are set correct!
for env_var, path in path_dict.items():
  if os.getenv(env_var) != path:
    print("Error:")
    print("Environment Variable {} is not set correctly!".format(env_var))
    print("Should be {}".format(path))
    print("Variable is {}".format(os.getenv(env_var)))
  make_if_dont_exist(path, overwrite=False)

print("If No Error Occured Continue Forward. =)")


os.chdir(path_dict["RAW_DATA_PATH"])
# Download the Hippocampus Dataset
# gdown.download('https://drive.google.com/uc?export=download&id=1L-22VV6J8O6afTSOQuQKFiH-tblxb_TW',"Hippocampus" ,quiet=False)

# # Download the Prostate Dataset
# gdown.download('https://drive.google.com/uc?export=download&id=1L-4D5szfpo7eO639TBmnukw9y_X9h5Yc',"Prostate" ,quiet=False)
# os.chdir(base_dir)

# print("Data should be located in folder: {}".format(path_dict["RAW_DATA_PATH"]))
# assert os.path.isfile(os.path.join(path_dict["RAW_DATA_PATH"], "Task04_Hippocampus.zip")) # check whether the file is correctly downloaded
# assert os.path.isfile(os.path.join(path_dict["RAW_DATA_PATH"], "Task05_Prostate.zip")) # check whether the file is correctly downloaded
