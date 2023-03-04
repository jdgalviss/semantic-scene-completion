import MinkowskiEngine as Me
import torch
import torch.nn as nn
import numpy as np
from configs import config
from .sparse_seg_net import SparseSegNet
from .ssc_head import SSCHead
from .model_utils import VoxelPooling
from structures import collect


class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__()
        self.seg_model = SparseSegNet()
        self.voxelpool = VoxelPooling()
        self.ssc_model = SSCHead(num_output_channels=config.MODEL.NUM_OUTPUT_CHANNELS, unet_features=config.MODEL.NUM_INPUT_FEATURES)


    def forward(self, batch, seg_weights=None, compl_weights=None):
        losses, results = {}, {}
        seg_feat = None
        if config.MODEL.SEG_HEAD:
            complet_invoxel_features = collect(batch, "complet_invoxel_features")
            voxel_centers = collect(batch, "voxel_centers")
            seg_out, seg_feat, loss = self.seg_model(coords=collect(batch, "seg_coords"), 
                                               feat=collect(batch, "seg_features"),
                                               label=collect(batch, "seg_labels"),
                                               weights=seg_weights)

            # pool feature vector before passing it as input to completion network
            seg_feat = torch.cat([seg_out,seg_feat],dim=1)
            seg_feat = self.voxelpool(invoxel_xyz=complet_invoxel_features[:, :, :-1],
                     invoxel_map=complet_invoxel_features[:, :, -1].long(),
                     src_feat=seg_feat,
                     voxel_center=voxel_centers)
            loss = {"pc_seg": loss}
            seg_out = {"pc_seg": seg_out}
            losses.update(loss)
            results.update(seg_out)
        loss, result = self.ssc_model(batch, seg_feat, compl_weights)
        losses.update(loss)
        results.update(result)
        return results, losses
    
    def inference(self,batch):
        results = {}

        if config.MODEL.SEG_HEAD:
            complet_invoxel_features = collect(batch, "complet_invoxel_features")
            voxel_centers = collect(batch, "voxel_centers")
            seg_out, seg_feat = self.seg_model.inference(coords=collect(batch, "seg_coords"), 
                                               feat=collect(batch, "seg_features"))

            # pool feature vector before passing it as input to completion network
            seg_feat = torch.cat([seg_out,seg_feat],dim=1)
            seg_feat = self.voxelpool(invoxel_xyz=complet_invoxel_features[:, :, :-1],
                     invoxel_map=complet_invoxel_features[:, :, -1].long(),
                     src_feat=seg_feat,
                     voxel_center=voxel_centers)
            seg_out = {"pc_seg": seg_out}
            results.update(seg_out)
        result = self.ssc_model.inference(batch, seg_feat)
        results.update(result)
        return results