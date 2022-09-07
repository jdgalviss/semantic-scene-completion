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

config_file = os.path.join('configs/semantic-kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))
remapdict = kitti_config["learning_map"]

SPLIT_SEQUENCES = {
    # "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    "train": ["00"],

    "valid": ["08"],
    "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
}

SPLIT_FILES = {
    "train": [".bin", ".label", ".invalid", ".occluded"],
    "valid": [".bin", ".label", ".invalid", ".occluded"],
    "test": [".bin"]
}

EXT_TO_NAME = {".bin": "input", ".label": "label", ".invalid": "invalid", ".occluded": "occluded"}
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
    def __init__(self, config, split="train", augment=False):
        """ Load data from given dataset directory. """

        self.config = config
        self.augment = augment
        self.files = {}
        self.filenames = []
        self.seg_path = config.GENERAL.DATASET_DIR
        # Create dictionary where keys are each ones of the extensions present in the split
        for ext in SPLIT_FILES[split]:
            self.files[EXT_TO_NAME[ext]] = []

        self.label_to_names = {0: 'car', 1: 'bicycle', 2: 'motorcycle', 3: 'truck',
                               4: 'other-vehicle', 5: 'person', 6: 'bicyclist', 7: 'motorcyclist',
                               8: 'road', 9: 'parking', 10: 'sidewalk', 11: 'other-ground', 12: 'building',
                               13: 'fence', 14: 'vegetation', 15: 'trunk', 16: 'terrain', 17: 'pole',
                               18: 'traffic-sign'}

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
                self.files[EXT_TO_NAME[ext]].extend(comletion_data)

            self.filenames.extend(
                sorted([(sequence, os.path.splitext(f)[0]) for f in files if f.endswith(SPLIT_FILES[split][0])]))

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
            # print(k, len(v))
            assert (len(v) == self.num_files)

        if split == 'train':
            seg_num_per_class = np.array(config.TRAIN.SEG_NUM_PER_CLASS)
            complt_num_per_class = np.array(config.TRAIN.COMPLT_NUM_PER_CLASS)

            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            self.seg_labelweights = np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0)
            compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
            self.compl_labelweights = np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)
        else:
            self.compl_labelweights = torch.Tensor(np.ones(20) * 3)
            self.seg_labelweights = torch.Tensor(np.ones(19))
            self.compl_labelweights[0] = 1

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        """ fill dictionary with available data for given index. """
        '''Load Completion Data'''
        completion_collection = {}
        if self.augment:
            # stat = np.random.randint(0,6)
            stat = np.random.randint(0,4)
        else:
            stat = 0 # set 0 with no augment
        completion_collection['stat'] = stat

        # read raw data and unpack (if necessary)
        for typ in self.files.keys():
            if typ == "label":
                scan_data = np.fromfile(self.files[typ][t], dtype=np.uint16)
                scan_data = self.comletion_remap_lut[scan_data]
            else:
                scan_data = unpack(np.fromfile(self.files[typ][t], dtype=np.uint8))
            scan_data = scan_data.reshape(self.config.COMPLETION.FULL_SCALE)
            scan_data = data_augmentation(torch.Tensor(scan_data).unsqueeze(0), stat)
            # turn in actual voxel grid representation.
            completion_collection[typ] = scan_data

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
        label = label[idxs]

        coords = torch.Tensor(coords).long()
        feature = torch.Tensor(feature)
        label = torch.Tensor(label)

        return coords, label, feature, idxs


def data_augmentation(t, state, inverse=False):
    assert t.dim() == 4, 'input dimension should be 4!'
    if state == 1:
        aug_t = t.flip([1])
    elif state == 2:
        aug_t = t.flip([2])
    # elif state == 3:
    #     k = 1 if not inverse else 3
    #     aug_t = t.rot90(k, [1, 2])
    # elif state == 4:
    #     aug_t = t.rot90(2, [1, 2])
    # elif state == 5:
    #     k = 3 if not inverse else 1
    #     aug_t = t.rot90(k, [1, 2])
    else:
        aug_t = t

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

