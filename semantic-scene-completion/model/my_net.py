import MinkowskiEngine as Me
import torch
import torch.nn as nn
import numpy as np
from configs import config
# from .sparse_seg_net import SparseSegNet
from .seg_2dpass_net import SparseSegNet2DPASS
# from .scpnet_priors import SCPNetPriorsModel
from .ssc_head import SSCHead
from .model_utils import VoxelPooling
from structures import collect
from torch.nn import functional as F
device = torch.device('cuda:0')

class MyModel(nn.Module):
    '''Model for semantic scene completion.
    Input pointclouds are processed with a segmentation model. Pointcloud features and pointcloud segmentation logits are passed
    to a semantic completion model (sparse generative)'''
    def __init__(self, is_teacher=False, **kwargs):
        '''
        :param is_teacher: if True, the model is used as a multi-pointcloud teacher model for distillation
        '''
        super(MyModel, self).__init__()
        self.complet_sigma = nn.Parameter(torch.Tensor(6).uniform_(0.2, 1), requires_grad=True) # Sigma for uncertainty loss
        self.is_teacher = is_teacher
        self.suffix = "_multi" if self.is_teacher else "" # suffix used for data field names

        if config.MODEL.SEG_HEAD:
            if config.SEGMENTATION.SEG_MODEL == "2DPASS":
                self.seg_model = SparseSegNet2DPASS().cuda() # pretrained 2DPASS model
                self.voxelpool = VoxelPooling()
                print("2DPASS model")
            elif config.SEGMENTATION.SEG_MODEL == "scpnet":
                self.seg_model = SCPNetPriorsModel() # SCPNet model
            else:
                self.seg_model = SparseSegNet() # Vanilla Sparse 3D UNet
            self.seg_sigma = nn.Parameter(torch.Tensor(1).uniform_(0.2, 1), requires_grad=True) # only relevant if segmentation model is also trained
            if not config.SEGMENTATION.TRAIN:
                self.seg_sigma.requires_grad = False
                self.seg_model.requires_grad = False
        else:
            self.seg_sigma = None
        # Sparse Generative Semantic Scene Completion model
        self.ssc_model = SSCHead(num_output_channels=config.MODEL.NUM_OUTPUT_CHANNELS, unet_features=config.MODEL.NUM_INPUT_FEATURES, suffix=self.suffix)


    def forward(self, batch, seg_weights=None, compl_weights=None):
        losses, results = {}, {}
        seg_feat = None
        if config.MODEL.SEG_HEAD:
            if config.SEGMENTATION.SEG_MODEL == "2DPASS":
                complet_invoxel_features = collect(batch, "complet_invoxel_features{}".format(self.suffix))
                voxel_centers = collect(batch, "voxel_centers{}".format(self.suffix))
                # Segmentation model forward pass
                seg_out, seg_feat, loss = self.seg_model(coords=collect(batch, "seg_coords{}".format(self.suffix)), 
                                                feat=collect(batch, "seg_features{}".format(self.suffix)),
                                                label=collect(batch, "seg_labels{}".format(self.suffix)),
                                                weights=seg_weights)
                if not config.SEGMENTATION.TRAIN:
                    seg_out = seg_out.detach()
                    seg_feat = seg_feat.detach()
                    loss = loss.detach()
                # voxelize feature vector before passing it as input to completion network
                seg_feat = torch.cat([seg_feat, seg_out],dim=1)
                seg_feat = self.voxelpool(invoxel_xyz=complet_invoxel_features[:, :, :-1],
                        invoxel_map=complet_invoxel_features[:, :, 3].long(),
                        src_feat=seg_feat,
                        voxel_center=voxel_centers)
                loss = {"pc_seg": loss}
                seg_out = {"pc_seg": seg_out}
                losses.update(loss)
                results.update(seg_out)
            elif config.SEGMENTATION.SEG_MODEL == "scpnet":
                train_pt_fea_ten = [collect(batch,"seg_features").type(torch.FloatTensor).to(device)]
                train_vox_ten = [collect(batch,"seg_coords")[:,1:].to(device)]
                with torch.no_grad():
                    coords_scpnet, features_scpnet = self.seg_model(train_pt_fea_ten, train_vox_ten, 1)
                seg_feat = (coords_scpnet, features_scpnet)
            
        # Semantic Scene Completion model forward pass
        loss, result, features = self.ssc_model(batch, seg_feat, compl_weights)
        losses.update(loss)
        results.update(result)
        return results, losses, features, [self.seg_sigma, self.complet_sigma]
    
    def inference(self,batch):
        '''Inference function for semantic scene completion (does not return loss).'''
        results = {}
        seg_feat = None
        if config.MODEL.SEG_HEAD:
            if config.SEGMENTATION.SEG_MODEL == "2DPASS":
                complet_invoxel_features = collect(batch, "complet_invoxel_features{}".format(self.suffix))
                voxel_centers = collect(batch, "voxel_centers{}".format(self.suffix))
                seg_out, seg_feat = self.seg_model.inference(coords=collect(batch, "seg_coords{}".format(self.suffix)), 
                                                feat=collect(batch, "seg_features{}".format(self.suffix)))
                seg_feat = torch.cat([seg_feat,seg_out],dim=1)
                seg_feat = self.voxelpool(invoxel_xyz=complet_invoxel_features[:, :, :-1],
                        invoxel_map=complet_invoxel_features[:, :, 3].long(),
                        src_feat=seg_feat,
                        voxel_center=voxel_centers)
                seg_out = {"pc_seg": seg_out}
                results.update(seg_out)
            elif config.SEGMENTATION.SEG_MODEL == "scpnet":
                train_pt_fea_ten = [collect(batch,"seg_features").type(torch.FloatTensor).to(device)]
                train_vox_ten = [collect(batch,"seg_coords")[:,1:].to(device)]
                with torch.no_grad():
                    coords_scpnet, features_scpnet = self.seg_model(train_pt_fea_ten, train_vox_ten, 1)
                seg_feat = (coords_scpnet, features_scpnet)
        result = self.ssc_model.inference(batch, seg_feat)
        results.update(result)
        return results