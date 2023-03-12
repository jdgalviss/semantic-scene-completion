from typing import Tuple, Dict

import MinkowskiEngine as Me
import torch
import torch.nn as nn
from configs import config
# Type hints
ModuleResult = Tuple[Dict, Dict]


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for im in tensor:
            for ch, m, s in zip(im, self.mean, self.std):
                ch.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        return tensor

def sparse_cat_union(a: Me.SparseTensor, b: Me.SparseTensor):
    cm = a.coordinate_manager
    assert cm == b.coordinate_manager, "different coords_man"
    assert a.tensor_stride == b.tensor_stride, "different tensor_stride"

    zeros_cat_with_a = torch.zeros([a.F.shape[0], b.F.shape[1]], dtype=a.dtype).to(a.device)
    zeros_cat_with_b = torch.zeros([b.F.shape[0], a.F.shape[1]], dtype=a.dtype).to(a.device)

    feats_a = torch.cat([a.F, zeros_cat_with_a], dim=1)
    feats_b = torch.cat([zeros_cat_with_b, b.F], dim=1)

    new_a = Me.SparseTensor(
        features=feats_a,
        coordinates=a.C,
        coordinate_manager=cm,
        tensor_stride=a.tensor_stride,
    )

    new_b = Me.SparseTensor(
        features=feats_b,
        coordinates=b.C,
        coordinate_manager=cm,
        tensor_stride=a.tensor_stride,
    )

    return new_a + new_b


def get_sparse_values(tensor: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    values = tensor[coordinates[:, 0], :, coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]]
    return values


def find_location_in_coordinates(coordinates, location):
    coord_mask = (coordinates[:, 1] == location[0]).bool() & (coordinates[:, 2] == location[1]).bool() & (
                coordinates[:, 3] == location[2]).bool()
    coord_index = coord_mask.nonzero()
    return coord_index


def thicken_grid(grid, grid_dims, frustum_mask):
    offsets = torch.nonzero(torch.ones(3, 3, 3)).long()
    locs_grid = grid.nonzero(as_tuple=False)
    locs = locs_grid.unsqueeze(1).repeat(1, 27, 1)
    locs += offsets
    locs = locs.view(-1, 3)
    mask_x = (locs[:, 0] >= 0) & (locs[:, 0] < grid_dims[0])
    mask_y = (locs[:, 1] >= 0) & (locs[:, 1] < grid_dims[1])
    mask_z = (locs[:, 2] >= 0) & (locs[:, 2] < grid_dims[2])
    locs = locs[mask_x & mask_y & mask_z]

    thicken = torch.zeros(grid_dims, dtype=torch.bool)
    thicken[locs[:, 0], locs[:, 1], locs[:, 2]] = True
    # frustum culling
    thicken = thicken & frustum_mask

    return thicken

class VoxelPooling(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.fuse_k = 20 
        self.pooling_mode = 'mean'
        if self.fuse_k > 1:
            self.relation_w = nn.Conv1d(10, config.MODEL.NUM_INPUT_FEATURES, 1)

    @staticmethod
    def index_feat(feature, index):
        device = index.device
        N, K = index.shape
        mask = None
        if K > 1:
            group_first = index[:, 0].view((N, 1)).repeat([1, K]).to(device)
            mask = index == 0
            index[mask] = group_first[mask]
        flat_index = index.reshape((N * K,))
        selected_feat = feature[flat_index, ]
        if K > 1:
            selected_feat = selected_feat.reshape((N, K, -1))
        else:
            selected_feat = selected_feat.reshape((N, -1))
        return selected_feat, mask

    @staticmethod
    def relation_position(group_xyz, center_xyz):
        K = group_xyz.shape[1]
        tile_center = center_xyz.unsqueeze(1).repeat([1, K, 1])
        offset = group_xyz - tile_center
        dist = torch.norm(offset, p=None, dim=-1, keepdim=True)
        relation = torch.cat([offset, tile_center, group_xyz, dist], -1)
        return relation

    def forward(self, invoxel_xyz, invoxel_map, src_feat, voxel_center=None):
        device = src_feat.device
        voxel2point_map = invoxel_map[:, :self.fuse_k].long()
        features, mask = self.index_feat(src_feat, voxel2point_map)  # [N, K, m]

        if self.fuse_k > 1:
            if self.pooling_mode == 'mean':
                features = features.mean(1)
            elif self.pooling_mode == 'max':
                features = features.max(1)[0]
            elif self.pooling_mode == 'relation':
                '''Voxel relation learning'''
                invoxel_xyz = invoxel_xyz[:, :self.fuse_k].to(device)
                N, K, _ = invoxel_xyz.shape
                group_first = invoxel_xyz[:, 0].view((N, 1, 3)).repeat([1, K, 1]).to(device)
                invoxel_xyz[mask, :] = group_first[mask, :]
                relation = self.relation_position(invoxel_xyz, voxel_center.to(device))
                group_w = self.relation_w(relation.permute(0, 2, 1))
                features = features.permute(0, 2, 1)
                features *= group_w
                features = torch.mean(features, 2)

        return features