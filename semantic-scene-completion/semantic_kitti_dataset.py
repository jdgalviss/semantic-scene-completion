# *_*coding:utf-8 *_*
"""
Author: Xu Yan
File: kitti_dataset.py
@time: 2020/8/12 22:03
"""
import os
import numpy as np
from utils import laserscan
import yaml
from torch.utils.data import Dataset
import torch
import math
from spconv.pytorch.utils import PointToVoxel
from structures import FieldList
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R
from utils.transforms import get_2d_input, get_bev
from configs import config

config_file = os.path.join('configs/semantic-kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))
remapdict = kitti_config["learning_map"]

SPLIT_SEQUENCES = {
    "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    "valid": ["08"],
    # "test": ["08"],
    "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
    "trainval": ["00"],
}


SPLIT_FILES = {
    "train": [".bin", ".label", ".label_128", ".label_64", ".invalid", ".invalid_64", ".invalid_128", ".occluded"],
    "valid": [".bin", ".label", ".label_128", ".label_64", ".invalid", ".invalid_64", ".invalid_128", ".occluded"],
    "test": [".bin"],
    "trainval": [".bin", ".label", ".label_128", ".label_64", ".invalid", ".invalid_64", ".invalid_128",  ".occluded"],

}

IGNORE_PER_LEVEL = {
    "64": ["label", "invalid", "labe_128", "invalid_128"],
    "128": ["label", "invalid"],
    "256": [],
    "FULL": [],
}

SCALE_PER_LEVEL = {
    "64": "_64",
    "128": "_128",
    "256": "", #aka 256
    "FULL": "",  #aka 256
}

EXT_TO_NAME = {".bin": "input", ".label": "label", ".label_128": "label_128", ".label_64": "label_64", ".invalid": "invalid", ".invalid_128": "invalid_128", ".invalid_64": "invalid_64", ".occluded": "occluded"}
scan = laserscan.SemLaserScan(nclasses=20, sem_color_dict=kitti_config['color_map'])


def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

def get_labelweights():
    ''' given a labelweights vector, make a labelweights matrix.  '''
    seg_num_per_class = np.array(config.TRAIN.SEG_NUM_PER_CLASS)
    complt_num_per_class = np.array(config.TRAIN.COMPLT_NUM_PER_CLASS)

    seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
    seg_labelweights = np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0)
    seg_labelweights = 1.0*seg_labelweights/np.linalg.norm(seg_labelweights)

    compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
    compl_labelweights = np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)
    compl_labelweights = 1.0*compl_labelweights/np.linalg.norm(compl_labelweights)

    return torch.Tensor(seg_labelweights), torch.Tensor(compl_labelweights)
    
class SemanticKITTIDataset(Dataset):
    def __init__(self, split="train", augment=False, do_overfit=False, num_samples_overfit=1):
        """ Load data from given dataset directory. """
        self.split = split
        self.augment = augment
        self.files = {}
        self.filenames = []
        self.seg_path = config.GENERAL.DATASET_DIR
        if config.GENERAL.OVERFIT:
            SPLIT_SEQUENCES["train"] = ["00"]
        # Create dictionary where keys are each ones of the extensions present in the split
        for ext in SPLIT_FILES[split]:
            self.files[EXT_TO_NAME[ext]] = []
        self.label_to_names = {0: 'unlabeled', 1: 'car', 2: 'bicycle', 3: 'motorcycle', 4: 'truck',
                               5: 'other-vehicle', 6: 'person', 7: 'bicyclist', 8: 'motorcyclist',
                               9: 'road', 10: 'parking', 11: 'sidewalk', 12: 'other-ground', 13: 'building',
                               14: 'fence', 15: 'vegetation', 16: 'trunk', 17: 'terrain', 18: 'pole',
                               19: 'traffic-sign', 20: 'other-object', 21: 'other-object'}
        # Iterate over all sequences present in split
        self.samples_per_split = {}
        self.all_poses = {}

        for sequence in SPLIT_SEQUENCES[split]:
            # Form path to voxels in split
            complete_path = os.path.join(config.GENERAL.DATASET_DIR, "sequences", sequence, "voxels")
            if not os.path.exists(complete_path): raise RuntimeError("Voxel directory missing: " + complete_path)

            files = os.listdir(complete_path)

            for ext in SPLIT_FILES[split]:
                # Obtain paths for all files with given extansion and sort
                comletion_data = sorted([os.path.join(complete_path, f) for f in files if f.endswith(ext)])
                if len(comletion_data) == 0: raise RuntimeError("Missing data for " + EXT_TO_NAME[ext])
                # Add paths to dictionary
                if(do_overfit):
                    completion_data_list = []
                    # Choose the samples to overfit
                    step = math.floor(len(comletion_data)/num_samples_overfit)
                    idxs = [i*step for i in range(num_samples_overfit)]
                    for i in idxs:
                        completion_data_list.append(comletion_data[i])
                    comletion_data = completion_data_list

                self.files[EXT_TO_NAME[ext]].extend(comletion_data)

            self.filenames.extend(
                sorted([(sequence, os.path.splitext(f)[0]) for f in files if f.endswith(SPLIT_FILES[split][0])]))
            
            ## poses
            poses_split = []
            poses_path = complete_path.replace('voxels','poses.txt')
            # Read calib file
            calib_path = complete_path.replace('voxels','calib.txt')
            calib = self.read_calib(calib_path)
            Tr = calib['Tr']
            Tr_inv = np.linalg.inv(calib['Tr'])
            with open(poses_path, 'r') as f:
                for line in f.readlines():
                    if line == '\n':
                        break
                    pose = line.split()
                    pose = np.float32(pose).reshape(3,4)
                    pose = np.concatenate([np.array(pose),np.array([0.,0,0.,1.]).reshape(1,4)],axis=0)
                    pose = np.matmul(Tr_inv,np.matmul(pose,Tr))
                    poses_split.append(pose)
            self.samples_per_split.update({sequence: len(poses_split)})
            self.all_poses.update({sequence: np.array(poses_split)})
        
        if(do_overfit):
            filenames_list = []
            step = math.floor(len(self.filenames )/num_samples_overfit)
            idxs = [i*step for i in range(num_samples_overfit)]
            for i in idxs:
                filenames_list.append(self.filenames[i])
            self.filenames = filenames_list
        
        
        self.num_files = len(self.filenames)
        remapdict = kitti_config["learning_map"]
        # make lookup table for mapping
        maxkey = max(remapdict.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        seg_remap_lut = remap_lut - 1
        seg_remap_lut[seg_remap_lut == -1] = -100

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.
        self.comletion_remap_lut = remap_lut
        self.seg_remap_lut = seg_remap_lut

        # sanity check:
        for k, v in self.files.items():
            assert (len(v) == self.num_files)

        num_point_features = 8 if config.MODEL.USE_COORDS else 5
        self.voxel_generator = PointToVoxel(
            vsize_xyz=[config.COMPLETION.VOXEL_SIZE]*3,
            coors_range_xyz=config.COMPLETION.POINT_CLOUD_RANGE,
            num_point_features=num_point_features,
            max_num_points_per_voxel=20,
            max_num_voxels=256 * 256 * 32
        )

    def __len__(self):
        return self.num_files
    
    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out
    
    def points_in_range(self,xyz,labels,remissions):
        keep_idxs = np.where((xyz[:, 0] >= config.COMPLETION.POINT_CLOUD_RANGE[0]) & (xyz[:, 0] <= config.COMPLETION.POINT_CLOUD_RANGE[3]) &
                             (xyz[:, 1] >= config.COMPLETION.POINT_CLOUD_RANGE[1]) & (xyz[:, 1] <= config.COMPLETION.POINT_CLOUD_RANGE[4]) &
                             (xyz[:, 2] >= config.COMPLETION.POINT_CLOUD_RANGE[2]) & (xyz[:, 2] <= config.COMPLETION.POINT_CLOUD_RANGE[5]))[0]
        return xyz[keep_idxs],labels[keep_idxs],remissions[keep_idxs]

    def __getitem__(self, t):
        """ fill dictionary with available data for given index. """
        '''Load Completion Data'''
        # t=3502 460 31 1051
        # t = 1051
        # print(t)
        completion_collection = {}
        if self.augment:
            flip_mode = np.random.randint(0,4)
            rot_zyx=[np.random.uniform(config.TRAIN.ROT_AUG_Z[0], config.TRAIN.ROT_AUG_Z[1]), 
                    np.random.uniform(config.TRAIN.ROT_AUG_Y[0], config.TRAIN.ROT_AUG_Y[1]),
                    np.random.uniform(config.TRAIN.ROT_AUG_X[0], config.TRAIN.ROT_AUG_X[1])]
        else:
            flip_mode = 0 # set 0 with no augment
            rot_zyx=[0,0,0]
        completion_collection['flip_mode'] = flip_mode
        # flip_mode = 1
        # rot_zyx=[-20,-0.5,1]
        # rot_zyx=[0,0,0]

        # read raw data and unpack (if necessary)
        for typ in self.files.keys():
            if typ in IGNORE_PER_LEVEL[config.GENERAL.LEVEL]:
                continue
            if typ == "label" or typ == "label_128" or typ == "label_64":
                scan_data = np.fromfile(self.files[typ][t], dtype=np.uint16)            
            else:
                scan_data = unpack(np.fromfile(self.files[typ][t], dtype=np.uint8))
            
            if typ == "label":
                scan_data = self.comletion_remap_lut[scan_data]
            if typ == "label_128" or typ == "invalid_128":
                scan_data = np.int32(scan_data.reshape(config.COMPLETION.SECOND_SCALE))
            elif typ == "label_64" or typ == "invalid_64":
                scan_data = np.int32(scan_data.reshape(config.COMPLETION.THIRD_SCALE))
            else:
                scan_data = scan_data.reshape(config.COMPLETION.FULL_SCALE)
            levels = {"label":256, "label_128":128, "label_64":64, "invalid":256, "invalid_128":128, "invalid_64":64}
            try:
                level=levels[typ]
            except:
                level=-1
                
            scan_data = self.data_augmentation(torch.Tensor(scan_data).unsqueeze(0), flip_mode, rot_zyx, level)
            
            # turn in actual voxel grid representation.
            completion_collection[typ] = scan_data
        
        if self.split != 'test':
            '''Load Segmentation Data'''
            seg_point_name = self.seg_path + self.files['input'][t][self.files['input'][t].find('sequences'):].replace('voxels','velodyne')
            seg_label_name = self.seg_path + self.files['label'][t][self.files['label'][t].find('sequences'):].replace('voxels','labels')

            scan.open_scan(seg_point_name)
            scan.open_label(seg_label_name)
            remissions = scan.remissions
            xyz = scan.points
            label = scan.sem_label
            label = self.seg_remap_lut[label]
            
            if config.MODEL.DISTILLATION:
                split, idx = self.filenames[t]
                # split = int(split)
                idx = int(idx)

                extra_idxs  = [(idx+i) for i in range(1,config.MODEL.DISTILLATION_SAMPLES) if (idx+i)<(self.samples_per_split[split]-1)]
                xyz_multi = xyz.copy()
                xyz_multi_raw = xyz.copy()

                remissions_multi = remissions.copy()
                label_multi = label.copy()
                id_multi = np.zeros_like(label)
                T0 = self.all_poses[split][idx]
                count = 1
                for i in extra_idxs:
                    aux_point_name = seg_point_name.replace("{:06d}.bin".format(idx), "{:06d}.bin".format(i))
                    aux_label_name = seg_label_name.replace("{:06d}.".format(idx), "{:06d}.".format(i))
                    scan.open_scan(aux_point_name)
                    scan.open_label(aux_label_name) # TODO: remove to improve memory usage

                    remissions_aux = scan.remissions
                    xyz_aux = scan.points
                    label_aux = scan.sem_label
                    label_aux = self.seg_remap_lut[label_aux]
                    # xyz_aux, remissions_aux, label_aux = self.points_in_range(xyz_aux,remissions_aux,label_aux)
                    xyz_multi_raw = np.concatenate((xyz_multi_raw, xyz_aux), axis=0)
                    # Transform to the same coordinate system as the first frame using homogeneous transformations
                    Ti = self.all_poses[split][i]
                    # T = np.linalg.inv(np.matmul(T0, np.linalg.inv(Ti)))
                    T = np.matmul(np.linalg.inv(T0),Ti)
                    xyz_aux_h = np.concatenate([xyz_aux, np.ones((xyz_aux.shape[0], 1))], axis=1)
                    xyz_aux = np.matmul(xyz_aux_h, T.T)[:, :3]
                    xyz_multi = np.concatenate((xyz_multi, xyz_aux), axis=0)

                    remissions_multi = np.concatenate((remissions_multi, remissions_aux), axis=0)
                    label_multi = np.concatenate((label_multi, label_aux), axis=0)
                    id_multi = np.concatenate((id_multi, count*np.ones_like(label_aux)), axis=0)
                    count += 1

        else:
            seg_point_name = self.seg_path + self.files['input'][t][self.files['input'][t].find('sequences'):].replace('voxels','velodyne')
            scan.open_scan(seg_point_name)
            
            xyz = scan.points
            remissions = scan.remissions
            label = None

        # Augmentations Point cloud
        if self.augment:
            remissions += np.random.randn(*remissions.shape)*config.TRAIN.NOISE_LEVEL

            # Drop points randomly from pointcloud
            pc_drop_prob = np.random.uniform(low=0.0, high=config.TRAIN.RANDOM_PC_DROP_AUG)
            keep_idxs = np.random.uniform(size=xyz.shape[0]) < (1.0 - pc_drop_prob)
            xyz = xyz[keep_idxs]
            label = label[keep_idxs]
            remissions = remissions[keep_idxs]
            # rotate
            r = R.from_euler('zyx', rot_zyx, degrees=True)
            r = r.as_matrix()
            xyz = np.matmul(xyz,r)
            if flip_mode == 1 or flip_mode == 2:
                xyz[:,1] = -xyz[:,1]
            
            if config.MODEL.DISTILLATION:
                remissions_multi += np.random.randn(*remissions_multi.shape)*config.TRAIN.NOISE_LEVEL
                # Drop points randomly from pointcloud
                pc_drop_prob = np.random.uniform(low=0.0, high=config.TRAIN.RANDOM_PC_DROP_AUG)
                keep_idxs = np.random.uniform(size=xyz_multi.shape[0]) < (1.0 - pc_drop_prob)
                xyz_multi = xyz_multi[keep_idxs]
                xyz_multi_raw = xyz_multi_raw[keep_idxs]
                label_multi = label_multi[keep_idxs]
                remissions_multi = remissions_multi[keep_idxs]
                id_multi = id_multi[keep_idxs]
                # Add translations
                translation = (np.random.normal(size=xyz_multi.shape))*0.04
                mask = np.random.uniform(size=translation.shape) < (1.0 - config.TRAIN.RANDOM_TRANSLATION_PROB)
                translation[mask] = 0.0
                xyz_multi+=translation
                # rotate
                xyz_multi = np.matmul(xyz_multi,r)
                if flip_mode == 1 or flip_mode == 2:
                    xyz_multi[:,1] = -xyz_multi[:,1]
            
        '''Process Segmentation Data'''
        if config.MODEL.USE_COORDS:
            feature = np.concatenate([xyz, remissions.reshape(-1, 1)], 1)
        else:
            feature = remissions.reshape(-1, 1)

        # Translation augmentation (not applied to pointcloud features because it disrupts pretrained model)
        if self.augment:
            translation = (np.random.normal(size=xyz.shape))*0.04
            # mask = np.random.uniform(size=translation.shape) < (1.0 - config.TRAIN.RANDOM_TRANSLATION_PROB)
            # translation[mask] = 0.0
            xyz+=translation
            
        segmentation_collection = {}
        coords, label, feature, idxs, m, random1, random2 = self.process_seg_data(xyz, label, feature)
        segmentation_collection.update({
            'coords': coords,
            'label': label,
            'feature': feature,
        })

        if config.MODEL.DISTILLATION:
            if config.MODEL.USE_COORDS:
                feature_multi = np.concatenate([xyz_multi_raw, remissions_multi.reshape(-1, 1)], 1)
            else:
                feature_multi = remissions_multi.reshape(-1, 1)
            
            coords_multi, label_multi, feature_multi, idxs_multi , _, _, _= self.process_seg_data(xyz_multi, label_multi, feature_multi, m, random1, random2)
            segmentation_collection.update({
                'coords_multi': coords_multi,
                'feature_multi': feature_multi,
                'label_multi': label_multi,
            })
        
        if config.MODEL.DISTILLATION:
            segmentation_collection.update({'id_multi': id_multi[idxs_multi]})

        elif config.MODEL.MULTI_ONLY:
            segmentation_collection.update({'id_multi': id_multi[idxs]})

        '''Generate Alignment Data'''
        aliment_collection = {}
        xyz = xyz[idxs]
        pc = torch.from_numpy(np.concatenate([xyz, np.arange(len(xyz)).reshape(-1,1), feature],-1)) # [x,y,z,idx,remission]
        voxels, coords, num_points_per_voxel = self.voxel_generator(pc)

        features = torch.sum(voxels[:,:,-1], dim=1) / torch.sum(voxels[:,:,-1] != 0, dim=1).clamp(min=1).float()
        coords = coords[:, [2, 1, 0]]
        # coords[:, 2] += 1  # TODO SemanticKITTI will generate [256,256,31]
        features = features[coords[:,2] < 32] # clamp to 32
        coords = coords[coords[:,2] < 32,:] # clamp to 32
        voxel_centers = (torch.flip(coords,[-1]) + 0.5) 
        coords = coords.long()

        voxel_centers *= torch.Tensor(self.voxel_generator.vsize)
        voxel_centers += torch.Tensor(self.voxel_generator.coors_range[0:3])
        
        # Input/Output for 2D BEV prediction model
        intensity_voxels = torch.zeros((1,256,256,32))
        intensity_voxels[:,coords[:,0],coords[:,1],coords[:,2]] = features

        if self.split != 'test':
            bev_labels = get_bev(completion_collection['label{}'.format(SCALE_PER_LEVEL[config.GENERAL.LEVEL])])
            aliment_collection.update({'bev_labels': bev_labels})
            completion_collection.update({'frustum_mask': completion_collection['label_64'] != -1})
            completion_collection['label_64'][completion_collection['label_64']==-1] = 255
            if config.GENERAL.LEVEL == "128" or config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                completion_collection['label_128'][completion_collection['label_128']==-1] = 255
                if config.GENERAL.LEVEL == "FULL":
                    completion_collection['label'][completion_collection['label']==-1] = 255

        aliment_collection.update({
            'voxels': voxels,
            'coords': coords,
            'voxel_centers': voxel_centers,
            'num_points_per_voxel': num_points_per_voxel,
            'features': features,
        })

        if config.MODEL.DISTILLATION:
            xyz_multi = xyz_multi[idxs_multi]
            pc_multi = torch.from_numpy(np.concatenate([xyz_multi, np.arange(len(xyz_multi)).reshape(-1,1), feature_multi],-1)) # [x,y,z,idx,remission]
            voxels_multi, coords_multi, num_points_per_voxel_multi = self.voxel_generator(pc_multi)
            features_multi = torch.sum(voxels_multi[:,:,-1], dim=1) / torch.sum(voxels_multi[:,:,-1] != 0, dim=1).clamp(min=1).float()
            coords_multi = coords_multi[:, [2, 1, 0]]
            features_multi = features_multi[coords_multi[:,2] < 32] # clamp to 32
            coords_multi = coords_multi[coords_multi[:,2] < 32,:] # clamp to 32
            voxel_centers_multi = (torch.flip(coords_multi,[-1]) + 0.5)
            coords_multi = coords_multi.long()
            voxel_centers_multi *= torch.Tensor(self.voxel_generator.vsize)
            voxel_centers_multi += torch.Tensor(self.voxel_generator.coors_range[0:3])
            aliment_collection.update({
                'voxels_multi': voxels_multi,
                'coords_multi': coords_multi,
                'voxel_centers_multi': voxel_centers_multi,
                'num_points_per_voxel_multi': num_points_per_voxel_multi,
                'features_multi': features_multi,
            })            
        # if self.split == "test":
        #     return aliment_collection
        return self.filenames[t], completion_collection, aliment_collection, segmentation_collection

    def process_seg_data(self, xyz, label, feature, m=None, random1=None, random2=None):
        coords = np.ascontiguousarray(xyz - xyz.mean(0)) # TODO: check if this should be kept for multisample pc
        if m is None:
            m = np.eye(3) + np.random.randn(3, 3) * 0.1
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
            m *= config.SEGMENTATION.SCALE
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
            random1 = np.random.rand(3)
            random2 = np.random.rand(3)
        coords = np.matmul(coords, m)
        minimum = coords.min(0)
        M = coords.max(0)
        offset = - minimum + np.clip(config.SEGMENTATION.FULL_SCALE[1] - M + minimum - 0.001, 0, None) * random1 + np.clip(
            config.SEGMENTATION.FULL_SCALE[1] - M + minimum + 0.001, None, 0) * random2
        coords += offset
        idxs = (coords.min(1) >= 0) * (coords.max(1) < config.SEGMENTATION.FULL_SCALE[1])
        coords = coords[idxs]
        feature = feature[idxs]
        if label is not None:
            label = label[idxs]
            label = torch.Tensor(label)
        else:
            label = None

        coords = torch.Tensor(coords).long()
        feature = torch.Tensor(feature)

        return coords, label, feature, idxs, m, random1, random2

    def data_augmentation(self, t, flip_mode, rot_zyx=[0,0,0], level=64, inp = False, inverse=False):
        rot_zyx_flipped =  rot_zyx.copy()# TODO (juan.galvis): correction for unwanted flip
        rot_zyx_flipped = [-rot_zyx[0], -rot_zyx[1], -rot_zyx[2]]
    
        assert t.dim() == 4, 'input dimension should be 4!'
        
        # ROTATION
        if self.augment and not inp:
            locs = torch.nonzero(t+1) 
            values = t[locs[:,0],locs[:,1],locs[:,2],locs[:,3]]
            locs_aug = locs.float()
            # move y to the center
            locs_aug[:,2] -= float(level)/2.0

            #get rotation matrix
            r = R.from_euler('zyx', rot_zyx_flipped, degrees=True)
            r = torch.from_numpy(r.as_matrix()).float()
            # Transform locations
            locs_aug[:,1:] = torch.matmul(locs_aug[:,1:],r)
            # Inverse centering translation
            locs_aug[:,2] += float(level)/2.0
            # locs_aug[:,3] -= 2
            locs_aug = torch.round(locs_aug).long()
            # keep values inside bounds
            valid = torch.where((locs_aug[:,1]>=0) & (locs_aug[:,2]>=0 ) & (locs_aug[:,3]>=0) & (locs_aug[:,1]<=(level-1)) & (locs_aug[:,2]<=(level-1)) & (locs_aug[:,3]<=((level/8)-1)))
            locs_aug = locs_aug[valid[0]]
            # values = values[valid[0]]
            values = t[locs_aug[:,0],locs_aug[:,1],locs_aug[:,2],locs_aug[:,3]]
            locs_aug = locs[valid[0]]
            # go back to voxel volume
            aug_t = torch.ones_like(t)*(-1)
            aug_t[locs_aug[:,0],locs_aug[:,1],locs_aug[:,2],locs_aug[:,3]] = values
        else:
            aug_t = t

        # Flio: we only flip 50% of the time
        if flip_mode == 1:
            # aug_t = t.flip([2])
            aug_t = aug_t.flip([2])
        elif flip_mode == 2:
            aug_t = aug_t.flip([2])
        else:
            aug_t = aug_t
        return aug_t

# def sparse_tensor_augmentation(st, states):
#     spatial_shape = st.spatial_shape
#     batch_size = st.batch_size
#     t = st.dense()
#     channels = t.shape[1]
#     for b in range(batch_size):
#         t[b] = data_augmentation(t[b], states[b])
#     coords = torch.sum(torch.abs(t), dim=1).nonzero().type(torch.int32)
#     features = t.permute(0, 2, 3, 4, 1).reshape(-1, channels)
#     features = features[torch.sum(torch.abs(features), dim=1).nonzero(), :]
#     features = features.squeeze(1)
#     nst = spconv.SparseConvTensor(features.float(), coords.int(), spatial_shape, batch_size)

#     return nst

def tensor_augmentation(st, states):
    batch_size = st.shape[0]
    for b in range(batch_size):
        st[b] = data_augmentation(st[b], states[b])

    return st

def Merge(tbl):
    seg_coords = []
    seg_features = []
    seg_labels = []
    complet_coords = []
    complet_invalid = []
    complet_invalid_64 = []
    complet_invalid_128 = []
    voxel_centers = []
    complet_invoxel_features = []
    complet_labels = []
    complet_labels_64 = []
    complet_labels_128 = []
    complet_features = []
    bev_labels = []
    frustum_mask = []
    filenames = []
    input_vx = []
    stats = []
    offset = 0

    if config.MODEL.DISTILLATION:
        voxel_centers_multi = []
        complet_coords_multi = []
        complet_features_multi = []
        seg_coords_multi = []
        seg_features_multi = []
        seg_labels_multi = []
        complet_invoxel_features_multi = []
        offset_multi = 0

    for idx, example in enumerate(tbl):
        filename, completion_collection, aliment_collection, segmentation_collection = example
        '''File Name'''
        filenames.append(filename)

        '''Segmentation'''
        seg_coord = segmentation_collection['coords']
        seg_coords.append(torch.cat([torch.LongTensor(seg_coord.shape[0], 1).fill_(idx), seg_coord], 1))
        seg_labels.append(segmentation_collection['label'])
        seg_features.append(segmentation_collection['feature'])

        '''Completion'''
        input_vx.append(completion_collection['input'])
        frustum_mask.append(completion_collection['frustum_mask'])

        complet_labels_64.append(completion_collection['label_64'])
        complet_invalid_64.append(completion_collection['invalid_64'])
        if config.GENERAL.LEVEL == "128" or config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
            complet_labels_128.append(completion_collection['label_128'])
            complet_invalid_128.append(completion_collection['invalid_128'])
            if config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                complet_labels.append(completion_collection['label'])
                complet_invalid.append(completion_collection['invalid'])
        stats.append(completion_collection['flip_mode'])

        complet_coord = aliment_collection['coords']
        complet_coord = torch.cat([torch.Tensor(complet_coord.shape[0], 1).fill_(idx), complet_coord.float()], 1)
        complet_coords.append(complet_coord)
        voxel_centers.append(torch.Tensor(aliment_collection['voxel_centers']))
        complet_invoxel_feature = aliment_collection['voxels']
        complet_invoxel_feature[:, :, -2] += offset  # voxel-to-point mapping in the last column
        complet_invoxel_features.append(torch.Tensor(complet_invoxel_feature))
        offset += seg_coord.shape[0]
        complet_features.append(aliment_collection['features'])
        bev_labels.append(aliment_collection['bev_labels'])

        # Additional inputs for multi_pc model whren distillation is enabled
        if config.MODEL.DISTILLATION:
            seg_coord_multi = segmentation_collection['coords_multi']
            seg_coords_multi.append(torch.cat([torch.LongTensor(segmentation_collection["id_multi"]).unsqueeze(1), seg_coord_multi], 1))
            # seg_coords_multi.append(torch.cat([torch.LongTensor(seg_coord_multi.shape[0], 1).fill_(idx), seg_coord_multi], 1))
            seg_features_multi.append(segmentation_collection['feature_multi'])
            seg_labels_multi.append(segmentation_collection['label_multi'])

            complet_coord_multi = aliment_collection['coords_multi']
            complet_coord_multi = torch.cat([torch.Tensor(complet_coord_multi.shape[0], 1).fill_(idx), complet_coord_multi.float()], 1)
            complet_coords_multi.append(complet_coord_multi)
            complet_features_multi.append(aliment_collection['features_multi'])
            voxel_centers_multi.append(torch.Tensor(aliment_collection['voxel_centers_multi']))
            complet_invoxel_feature_multi = aliment_collection['voxels_multi']
            complet_invoxel_feature_multi[:, :, -2] += offset_multi  # voxel-to-point mapping in the last column
            complet_invoxel_features_multi.append(torch.Tensor(complet_invoxel_feature_multi))
            offset_multi += seg_coord_multi.shape[0]
    
    # Field List that will include all the dateafields
    complet_inputs = FieldList((320, 240), mode="xyxy") # TODO: parameters are irrelevant
    
    one = torch.ones([1])
    zero = torch.zeros([1])
    frustum_mask = torch.cat(frustum_mask, 0)
    # LEVEL 64
    complet_labels_64 = torch.cat(complet_labels_64, 0)
    complet_invalid_64 = torch.cat(complet_invalid_64, 0)
    invalid_locs_64 = torch.nonzero(complet_invalid_64[0])
    # complet_labels_64 = F.max_pool3d(complet_labels_128.float(), kernel_size=2, stride=2).int()
    # complet_occupancy_64 = F.max_pool3d(complet_occupancy_128.float(), kernel_size=2, stride=2)
    complet_labels_64[0, invalid_locs_64[:, 0], invalid_locs_64[:, 1], invalid_locs_64[:, 2]] = 255
    invalid_locs = torch.where(complet_labels_64 > 255)
    complet_labels_64[invalid_locs] = 255
    complet_occupancy_64 = torch.where(complet_labels_64 > 0, one, zero)
    complet_inputs.add_field("complet_occupancy_64", complet_occupancy_64)
    complet_inputs.add_field("complet_labels_64", complet_labels_64)

    # Level 128
    if config.GENERAL.LEVEL == "128" or config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
        complet_labels_128 = torch.cat(complet_labels_128, 0)
        complet_invalid_128 = torch.cat(complet_invalid_128, 0)
        invalid_locs_128 = torch.nonzero(complet_invalid_128[0])
        # complet_labels_128 = F.max_pool3d(complet_labels.float(), kernel_size=2, stride=2).int()
        complet_labels_128[0, invalid_locs_128[:, 0], invalid_locs_128[:, 1], invalid_locs_128[:, 2]] = 255
        invalid_locs = torch.where(complet_labels_128 > 255)
        complet_labels_128[invalid_locs] = 255
        complet_occupancy_128 = torch.where(complet_labels_128 > 0  , one, zero)
        # complet_occupancy_128 = torch.where(torch.logical_and(complet_labels_128 > 0, complet_labels_128 < 255), one, zero) # TODO: is invalid occupied or unoccupied?
        complet_inputs.add_field("complet_occupancy_128", complet_occupancy_128)
        complet_inputs.add_field("complet_labels_128", complet_labels_128)

        # Level 256
        if config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
            complet_labels = torch.cat(complet_labels, 0)
            complet_invalid = torch.cat(complet_invalid, 0) 
            invalid_locs = torch.nonzero(complet_invalid[0])
            complet_valid = complet_labels != 255
            # invalid locations
            complet_labels[0,invalid_locs[:,0], invalid_locs[:,1], invalid_locs[:,2]] = 255
            invalid_locs = torch.where(complet_labels > 255)
            complet_labels[invalid_locs] = 255
            complet_occupancy = torch.where(complet_labels > 0  , one, zero)
            complet_inputs.add_field("complet_labels_256", complet_labels)
            complet_inputs.add_field("complet_occupancy_256", complet_occupancy)
            complet_inputs.add_field("complet_valid", complet_valid)

    complet_inputs.add_field("bev_labels", torch.cat(bev_labels, 0))
    complet_inputs.add_field("frustum_mask", frustum_mask)

    if not config.MODEL.MULTI_ONLY: # If single scan model is being trained, we need the single scan inputs
        complet_inputs.add_field("seg_coords", torch.cat(seg_coords, 0))
        complet_inputs.add_field("seg_labels", torch.cat(seg_labels, 0))
        complet_inputs.add_field("seg_features", torch.cat(seg_features, 0))
        complet_inputs.add_field("complet_coords", torch.cat(complet_coords, 0).unsqueeze(0))
        complet_inputs.add_field("complet_features", torch.cat(complet_features, 0).unsqueeze(0))
        complet_inputs.add_field("voxel_centers", torch.cat(voxel_centers, 0))
        complet_inputs.add_field("complet_invoxel_features", torch.cat(complet_invoxel_features, 0))
    
    if config.MODEL.DISTILLATION: # If distillation is being used, we need the multi scan inputs
        complet_invoxel_features_multi = torch.cat(complet_invoxel_features_multi, 0)
        complet_inputs.add_field("complet_coords_multi", torch.cat(complet_coords_multi, 0).unsqueeze(0))
        complet_inputs.add_field("complet_features_multi", torch.cat(complet_features_multi, 0).unsqueeze(0))
        complet_inputs.add_field("voxel_centers_multi", torch.cat(voxel_centers_multi, 0).unsqueeze(0))
        complet_inputs.add_field("seg_coords_multi", torch.cat(seg_coords_multi, 0))
        complet_inputs.add_field("seg_features_multi", torch.cat(seg_features_multi, 0))
        complet_inputs.add_field("seg_labels_multi", torch.cat(seg_labels_multi, 0))
        complet_inputs.add_field("complet_invoxel_features_multi", complet_invoxel_features_multi)

    return filenames, complet_inputs, None, filenames

def MergeTest(tbl):
    complet_coords = []
    voxel_centers = []
    complet_invoxel_features = []
    complet_features = []
    seg_coords = []
    seg_features = []
    seg_labels = []
    filenames = []
    offset = 0
    input_vx = []

    for idx, example in enumerate(tbl):
        filename, completion_collection, aliment_collection, segmentation_collection = example
        '''File Name'''
        filenames.append(filename)

        '''Segmentation'''
        seg_coord = segmentation_collection['coords']
        if config.MODEL.MULTI_ONLY:
            seg_coords.append(torch.cat([torch.LongTensor(segmentation_collection["id_multi"]).unsqueeze(1), seg_coord], 1))
        else:
            seg_coords.append(torch.cat([torch.LongTensor(seg_coord.shape[0], 1).fill_(idx), seg_coord], 1))
        seg_labels.append(segmentation_collection['label'])
        seg_features.append(segmentation_collection['feature'])


        '''Completion'''
        complet_coord = aliment_collection['coords']
        
            
        complet_coords.append(torch.cat([torch.Tensor(complet_coord.shape[0], 1).fill_(idx), complet_coord.float()], 1))

        input_vx.append(completion_collection['input'])

        voxel_centers.append(torch.Tensor(aliment_collection['voxel_centers']))
        complet_invoxel_feature = aliment_collection['voxels']
        complet_invoxel_feature[:, :, -2] += offset  # voxel-to-point mapping in the last column
        complet_invoxel_features.append(torch.Tensor(complet_invoxel_feature))
        offset += seg_coord.shape[0]
        complet_features.append(aliment_collection['features'])

        
    complet_inputs = FieldList((320, 240), mode="xyxy") # TODO: parameters are irrelevant
    complet_inputs.add_field("seg_coords", torch.cat(seg_coords, 0))
    complet_inputs.add_field("seg_features", torch.cat(seg_features, 0))

    complet_inputs.add_field("complet_coords", torch.cat(complet_coords, 0).unsqueeze(0))
    complet_inputs.add_field("complet_features", torch.cat(complet_features, 0).unsqueeze(0))
    complet_inputs.add_field("voxel_centers", torch.cat(voxel_centers, 0))
    complet_inputs.add_field("complet_invoxel_features", torch.cat(complet_invoxel_features, 0))


    # filenames = None
    return filenames, complet_inputs, None, filenames
