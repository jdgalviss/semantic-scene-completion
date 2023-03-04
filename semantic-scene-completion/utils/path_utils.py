import glob
import shutil
import os
from yacs.config import CfgNode as Node

def create_new_experiment_folder(folder_name: str) -> str:
    """
    Create a new experiment file name.
    """
    files = glob.glob(folder_name + "/*")
    if len(files)>0:
        max_file = max(files)
    else:
        max_file = "/000"
    experiment_dir = folder_name + "/{:03d}".format(int(max_file.split('/')[-1].split('_')[0])+1)
    os.mkdir(experiment_dir)
    return experiment_dir

def save_config(config: Node, save_path: str) -> None:
    """
    Save configuration file to experiment directory.
    """
    with open(save_path + "/config.yaml", "w") as f:
        f.write(config.dump())
        
    shutil.copyfile("train.py", save_path + "/train.py")
    shutil.copyfile("semantic_kitti_dataset.py", save_path + "/semantic_kitti_dataset.py")
    shutil.copyfile("model/hybrid_unet.py", save_path + "/hybrid_unet.py")
    shutil.copyfile("model/sparse_seg_net.py", save_path + "/sparse_seg_net.py")
    shutil.copyfile("model/ssc_head.py", save_path + "/ssc_head.py")
    shutil.copyfile("model/my_net.py", save_path + "/my_net.py")
