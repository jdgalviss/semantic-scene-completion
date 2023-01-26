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

config_file = os.path.join('configs/semantic-kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))
remapdict = kitti_config["learning_map"]

SPLIT_SEQUENCES = {
    "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    "valid": ["08"],
    "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
    "trainval": ["00"],

}

SPLIT_FILES = {
    "train": [".bin", ".label", ".label_128", ".label_64", ".invalid", ".invalid_64", ".invalid_128", ".occluded"],
    "valid": [".bin", ".label", ".label_128", ".label_64", ".invalid", ".invalid_64", ".invalid_128", ".occluded"],
    "test": [".bin"],
    "trainval": [".bin", ".label", ".label_128", ".label_64", ".invalid", ".invalid_64", ".invalid_128",  ".occluded"],

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

class SemanticKITTIDataset(Dataset):
    def __init__(self, config, split="train", augment=False, do_overfit=False, num_samples_overfit=1):
        """ Load data from given dataset directory. """
        self.split = split
        self.config = config
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
        if split == 'train':
            seg_num_per_class = np.array(config.TRAIN.SEG_NUM_PER_CLASS)
            complt_num_per_class = np.array(config.TRAIN.COMPLT_NUM_PER_CLASS)

            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            self.seg_labelweights = np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0)
            compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
            self.compl_labelweights = np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)
            # self.compl_labelweights = np.ones_like(self.compl_labelweights)
            
            
            self.compl_labelweights =1.0*self.compl_labelweights/np.linalg.norm(self.compl_labelweights)


            # self.compl_labelweights[1] = np.amax(self.compl_labelweights)
        else:
            self.compl_labelweights = torch.Tensor(np.ones(20) * 3)
            self.seg_labelweights = torch.Tensor(np.ones(19))
            self.compl_labelweights[0] = 1
        
        self.voxel_generator = PointToVoxel(
            vsize_xyz=[config.COMPLETION.VOXEL_SIZE]*3,
            coors_range_xyz=config.COMPLETION.POINT_CLOUD_RANGE,
            num_point_features=5, # or 3??
            max_num_points_per_voxel=20,
            max_num_voxels=256 * 256 * 32
        )

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        """ fill dictionary with available data for given index. """
        '''Load Completion Data'''
        completion_collection = {}
        if self.augment:
            # stat = np.random.randint(0,6)
            flip_mode = np.random.randint(0,4)
            rot_zyx=[np.random.uniform(self.config.TRAIN.ROT_AUG_Z[0], self.config.TRAIN.ROT_AUG_Z[1]), 
                    np.random.uniform(self.config.TRAIN.ROT_AUG_Y[0], self.config.TRAIN.ROT_AUG_Y[1]),
                    np.random.uniform(self.config.TRAIN.ROT_AUG_X[0], self.config.TRAIN.ROT_AUG_X[1])]
        else:
            flip_mode = 0 # set 0 with no augment
            rot_zyx=[0,0,0]
        completion_collection['flip_mode'] = flip_mode
        # flip_mode = 0
        # rot_zyx=[10,5,1]
        # rot_zyx=[0,0,0]


        # read raw data and unpack (if necessary)
        for typ in self.files.keys():
            if typ == "label" or typ == "label_128" or typ == "label_64":
                scan_data = np.fromfile(self.files[typ][t], dtype=np.uint16)            
            else:
                scan_data = unpack(np.fromfile(self.files[typ][t], dtype=np.uint8))
            
            if typ == "label":
                scan_data = self.comletion_remap_lut[scan_data]
            if typ == "label_128" or typ == "invalid_128":
                scan_data = np.int32(scan_data.reshape(self.config.COMPLETION.SECOND_SCALE))
            elif typ == "label_64" or typ == "invalid_64":
                scan_data = np.int32(scan_data.reshape(self.config.COMPLETION.THIRD_SCALE))
            else:
                scan_data = scan_data.reshape(self.config.COMPLETION.FULL_SCALE)
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

            if self.config.SEGMENTATION.USE_COORDS:
                feature = np.concatenate([xyz, remissions.reshape(-1, 1)], 1)
            else:
                feature = remissions.reshape(-1, 1)

            # Add noise augmentation to input data
            if self.augment:
                feature += np.random.randn(*feature.shape)*self.config.TRAIN.NOISE_LEVEL
        else:
            seg_point_name = self.seg_path + self.files['input'][t][self.files['input'][t].find('sequences'):].replace('voxels','velodyne')
            scan.open_scan(seg_point_name)
            
            xyz = scan.points
            remissions = scan.remissions
            if self.config.SEGMENTATION.USE_COORDS:
                feature = np.concatenate([xyz, remissions.reshape(-1, 1)], 1)
            else:
                feature = remissions.reshape(-1, 1)
            label = None


        '''Process Segmentation Data'''
        segmentation_collection = {}
        coords, label, feature, idxs = self.process_seg_data(xyz, label, feature)
        segmentation_collection.update({
            'coords': coords,
            'label': label,
            'feature': feature,
        })

        '''Generate Alignment Data'''
        aliment_collection = {}
        xyz = xyz[idxs]
        pc = torch.from_numpy(np.concatenate([xyz, np.arange(len(xyz)).reshape(-1,1), feature],-1))

        voxels, coords, num_points_per_voxel = self.voxel_generator(pc)
        voxel_centers = (torch.flip(coords,[-1]) + 0.5) 


        features = torch.sum(voxels[:,:,-1], dim=1) / torch.sum(voxels[:,:,-1] != 0, dim=1).clamp(min=1).float()
        coords = coords[:, [2, 1, 0]]
        coords[:, 2] += 1  # TODO SemanticKITTI will generate [256,256,31]
        coords = coords.long()

        # Apply Augmentation to input data
        # Create dense voxel grid of features so we can perform augmentation TODO(juan.galvis): this is slow.
        if self.augment:
            feature_voxels = torch.zeros((256,256,32)) # TODO (juan.galvis): Hardcoded
            feature_voxels[coords[:,0],coords[:,1],coords[:,2]] = features + 1.0 # TODO (juan.galvis): So we can use non_zero
            feature_voxels = self.data_augmentation(feature_voxels.unsqueeze(0), flip_mode,inp=True) # only flip transform
            # feature_voxels = feature_voxels.unsqueeze(0)
            # print("feature_voxels: ", feature_voxels.shape")

            coords = torch.nonzero(feature_voxels[0])
            features = feature_voxels[0,coords[:,0],coords[:,1],coords[:,2]] - 1.0
        
        # Rotation Augmentation Input Tensors
        if self.augment:
            values = features
            locs_aug = coords.float()
            # move y to the center
            locs_aug[:,1] -= 256./2.0
            #get rotation matrix
            r = R.from_euler('zyx', rot_zyx, degrees=True)
            r = torch.from_numpy(r.as_matrix()).float()
            # Transform locations
            locs_aug = torch.matmul(locs_aug,r)
            # Inverse centering translation
            locs_aug[:,1] += 256./2.0
            # Add translations
            translation = torch.randint(-1,2,locs_aug.shape)
            mask = torch.rand(translation.shape) < (1.0 - self.config.TRAIN.RANDOM_TRANSLATION_PROB)
            translation[mask] = 0
            locs_aug+=translation
            
            # check inside bounds
            valid = torch.where((locs_aug[:,0]>=0) & (locs_aug[:,1]>=0 ) & (locs_aug[:,2]>=0) & (locs_aug[:,0]<=255) & (locs_aug[:,1]<=255) & (locs_aug[:,2]<=31))
            locs_aug = locs_aug[valid[0]]
            values = values[valid[0]]
            coords = locs_aug.long()
            features = values

        # print("coords", coords.shape)
        # print("features", features.shape)
        # print("coords: ", type(coords) )

        # print(voxel_centers.shape)
        voxel_centers *= torch.Tensor(self.voxel_generator.vsize)
        voxel_centers += torch.Tensor(self.voxel_generator.coors_range[0:3])
        aliment_collection.update({
            'voxels': voxels,
            'coords': coords,
            'voxel_centers': voxel_centers,
            'num_points_per_voxel': num_points_per_voxel,
            'features': features,
        })
        # if self.split == "test":
        #     return aliment_collection
        return self.filenames[t], completion_collection, aliment_collection, segmentation_collection

    def process_seg_data(self, xyz, label, feature):
        coords = np.ascontiguousarray(xyz - xyz.mean(0))
        m = np.eye(3) + np.random.randn(3, 3) * 0.1
        m[0][0] *= np.random.randint(0, 2) * 2 - 1
        m *= self.config.SEGMENTATION.SCALE
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        coords = np.matmul(coords, m)

        m = coords.min(0)
        M = coords.max(0)
        offset = - m + np.clip(self.config.SEGMENTATION.FULL_SCALE[1] - M + m - 0.001, 0, None) * np.random.rand(3) + np.clip(
            self.config.SEGMENTATION.FULL_SCALE[1] - M + m + 0.001, None, 0) * np.random.rand(3)
        coords += offset
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.config.SEGMENTATION.FULL_SCALE[1])
        coords = coords[idxs]
        feature = feature[idxs]
        if label is not None:
            label = label[idxs]
            label = torch.Tensor(label)
        else:
            label = None

        coords = torch.Tensor(coords).long()
        feature = torch.Tensor(feature)

        return coords, label, feature, idxs


    def data_augmentation(self, t, flip_mode, rot_zyx=[0,0,0], level=64, inp = False, inverse=False):
        rot_zyx_flipped =  rot_zyx.copy()# TODO (juan.galvis): correction for unwanted flip
        assert t.dim() == 4, 'input dimension should be 4!'
        # FLIP (we only flip the 50% of the times)
        if flip_mode == 1:
            # aug_t = t.flip([2])
            aug_t = t.flip([2])
            rot_zyx_flipped[0] *= -1

        elif flip_mode == 2:
            aug_t = t.flip([2])
            rot_zyx_flipped[0] *= -1
        else:
            aug_t = t
        
        # ROTATION
        if self.augment and not inp:
            locs = torch.nonzero(t) 
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
            values = values[valid[0]]
            # go back to voxel volume
            aug_t = torch.zeros_like(t)
            aug_t[locs_aug[:,0],locs_aug[:,1],locs_aug[:,2],locs_aug[:,3]] = values
            # TODO (juan.galvis): correction for unwanted flip
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


    filenames = []
    offset = 0
    input_vx = []
    stats = []
    for idx, example in enumerate(tbl):
        filename, completion_collection, aliment_collection, segmentation_collection = example
        '''File Name'''
        filenames.append(filename)

        '''Segmentation'''
        seg_coord = segmentation_collection['coords']
        seg_coords.append(torch.cat([seg_coord, torch.LongTensor(seg_coord.shape[0], 1).fill_(idx)], 1))
        seg_labels.append(segmentation_collection['label'])
        seg_features.append(segmentation_collection['feature'])

        '''Completion'''
        complet_coord = aliment_collection['coords']
        complet_coords.append(torch.cat([torch.Tensor(complet_coord.shape[0], 1).fill_(idx), complet_coord.float()], 1))

        input_vx.append(completion_collection['input'])
        complet_labels.append(completion_collection['label'])
        complet_labels_128.append(completion_collection['label_128'])
        complet_labels_64.append(completion_collection['label_64'])
        complet_invalid.append(completion_collection['invalid'])
        complet_invalid_64.append(completion_collection['invalid_64'])
        complet_invalid_128.append(completion_collection['invalid_128'])

        stats.append(completion_collection['flip_mode'])

        voxel_centers.append(torch.Tensor(aliment_collection['voxel_centers']))
        complet_invoxel_feature = aliment_collection['voxels']
        complet_invoxel_feature[:, :, -2] += offset  # voxel-to-point mapping in the last column
        offset += seg_coord.shape[0]

        complet_features.append(aliment_collection['features'])
        complet_invoxel_features.append(torch.Tensor(complet_invoxel_feature))
    complet_invoxel_features = torch.cat(complet_invoxel_features, 0)
    # complet_features = torch.amax(complet_invoxel_features[:,:,-1], dim=1)
    complet_features = torch.cat(complet_features, 0)

    # seg_inputs = {'seg_coords': torch.cat(seg_coords, 0),
    #               'seg_labels': torch.cat(seg_labels, 0),
    #               'seg_features': torch.cat(seg_features, 0)
    #               }
    


    # complet_inputs = {'complet_coords': torch.cat(complet_coords, 0),
    #                   'complet_input': torch.cat(input_vx, 0),
    #                   'voxel_centers': torch.cat(voxel_centers, 0),
    #                   'complet_invalid': torch.cat(complet_invalid, 0),
    #                   'complet_labels': torch.cat(complet_labels, 0),
    #                   'state': stats,
    #                   'complet_invoxel_features': torch.cat(complet_invoxel_features, 0)
    #                   }
    one = torch.ones([1])
    zero = torch.zeros([1])
    complet_labels = torch.cat(complet_labels, 0)
    complet_labels_128 = torch.cat(complet_labels_128, 0)
    complet_labels_64 = torch.cat(complet_labels_64, 0)
    complet_invalid = torch.cat(complet_invalid, 0) 
    complet_invalid_128 = torch.cat(complet_invalid_128, 0)
    complet_invalid_64 = torch.cat(complet_invalid_64, 0)
    complet_coords = torch.cat(complet_coords, 0)
    invalid_locs = torch.nonzero(complet_invalid[0])
    invalid_locs_128 = torch.nonzero(complet_invalid_128[0])
    invalid_locs_64 = torch.nonzero(complet_invalid_64[0])

    
    # input_vx = torch.cat(input_vx, 0)    
    # input_coords = torch.nonzero(input_vx)

     # invalid locations
    complet_labels[0,invalid_locs[:,0], invalid_locs[:,1], invalid_locs[:,2]] = 255
    invalid_locs = torch.where(complet_labels > 255)
    complet_labels[invalid_locs] = 255
    complet_occupancy = torch.where(torch.logical_and(complet_labels > 0, complet_labels < 255), one, zero) # TODO: is invalid occupied or unoccupied?
    # complet_occupancy = torch.where(complet_labels > 0  , one, zero)

    
    # complet_labels_128 = F.max_pool3d(complet_labels.float(), kernel_size=2, stride=2).int()
    # complet_occupancy_128 = F.max_pool3d(complet_occupancy.float(), kernel_size=2, stride=2)
    complet_labels_128[0, invalid_locs_128[:, 0], invalid_locs_128[:, 1], invalid_locs_128[:, 2]] = 255
    invalid_locs = torch.where(complet_labels_128 > 255)
    complet_labels_128[invalid_locs] = 255
    # complet_occupancy_128 = torch.where(complet_labels_128 > 0  , one, zero)
    complet_occupancy_128 = torch.where(torch.logical_and(complet_labels_128 > 0, complet_labels_128 < 255), one, zero) # TODO: is invalid occupied or unoccupied?


    # complet_labels_64 = F.max_pool3d(complet_labels_128.float(), kernel_size=2, stride=2).int()
    # complet_occupancy_64 = F.max_pool3d(complet_occupancy_128.float(), kernel_size=2, stride=2)
    complet_labels_64[0, invalid_locs_64[:, 0], invalid_locs_64[:, 1], invalid_locs_64[:, 2]] = 255
    invalid_locs = torch.where(complet_labels_64 > 255)
    complet_labels_64[invalid_locs] = 255
    # complet_occupancy_64 = torch.where(complet_labels_64 > 0, one, zero)
    complet_occupancy_64 = torch.where(torch.logical_and(complet_labels_64 > 0, complet_labels_64 < 255), one, zero) # TODO: is invalid occupied or unoccupied?




    complet_inputs = FieldList((320, 240), mode="xyxy") # TODO: parameters are irrelevant
    complet_inputs.add_field("complet_coords", complet_coords.unsqueeze(0))
    complet_inputs.add_field("complet_invalid", complet_invalid)
    complet_inputs.add_field("complet_labels_256", complet_labels)
    complet_inputs.add_field("complet_occupancy_256", complet_occupancy)
    complet_inputs.add_field("complet_labels_128", complet_labels_128)
    complet_inputs.add_field("complet_occupancy_128", complet_occupancy_128)
    complet_inputs.add_field("complet_labels_64", complet_labels_64)
    complet_inputs.add_field("complet_occupancy_64", complet_occupancy_64)
    # complet_inputs.add_field("seg_coords", torch.cat(seg_coords, 0).unsqueeze(0))
    # complet_inputs.add_field("seg_labels", torch.cat(seg_labels, 0).unsqueeze(0))
    # complet_inputs.add_field("seg_features", torch.cat(seg_features, 0).transpose(0,1))
    complet_inputs.add_field("complet_features", complet_features.unsqueeze(0))
    # complet_inputs.add_field("input_coords", input_coords.unsqueeze(0))


    # complet_inputs.add_field("complet_invoxel_features", torch.cat(complet_invoxel_features, 0).unsqueeze(0))


    # del seg_inputs, completion_collection, filenames
    # seg_inputs = None
    # completion_collection = None
    # filenames = None
    return filenames, complet_inputs, None, filenames

def MergeTest(tbl):
    complet_coords = []
    voxel_centers = []
    complet_invoxel_features = []
    complet_features = []
    filenames = []
    offset = 0
    input_vx = []
    for idx, example in enumerate(tbl):
        filename, completion_collection, aliment_collection, _ = example
        '''File Name'''
        filenames.append(filename)


        '''Completion'''
        complet_coord = aliment_collection['coords']
        complet_coords.append(torch.cat([torch.Tensor(complet_coord.shape[0], 1).fill_(idx), complet_coord.float()], 1))

        input_vx.append(completion_collection['input'])

        voxel_centers.append(torch.Tensor(aliment_collection['voxel_centers']))
        complet_invoxel_feature = aliment_collection['voxels']
        complet_invoxel_feature[:, :, -2] += offset  # voxel-to-point mapping in the last column

        complet_features.append(aliment_collection['features'])
        complet_invoxel_features.append(torch.Tensor(complet_invoxel_feature))
    complet_invoxel_features = torch.cat(complet_invoxel_features, 0)
    complet_features = torch.cat(complet_features, 0)
    complet_coords = torch.cat(complet_coords, 0)

    complet_inputs = FieldList((320, 240), mode="xyxy") # TODO: parameters are irrelevant
    complet_inputs.add_field("complet_coords", complet_coords.unsqueeze(0))
    complet_inputs.add_field("complet_features", complet_features.unsqueeze(0))
    # filenames = None
    return filenames, complet_inputs, None, filenames
