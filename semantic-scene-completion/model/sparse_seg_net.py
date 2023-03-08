import MinkowskiEngine as Me
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from configs import config
from .sparse_unet import MinkUNet14E

device = torch.device("cuda:0")

class SparseSegNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        input_dim = 4 if config.MODEL.USE_COORDS else 1
        self.sparse_model = nn.Sequential(
            Me.MinkowskiConvolution(input_dim, config.MODEL.NUM_INPUT_FEATURES, kernel_size=3, bias=True, stride=1, dimension=3),
            Me.MinkowskiInstanceNorm(config.MODEL.NUM_INPUT_FEATURES),
            Me.MinkowskiReLU(inplace=True),
            MinkUNet14E(in_channels=config.MODEL.NUM_INPUT_FEATURES, out_channels=config.MODEL.NUM_INPUT_FEATURES, D=3),
            Me.MinkowskiInstanceNorm(config.MODEL.NUM_INPUT_FEATURES),
            Me.MinkowskiReLU(inplace=True),
        )
        self.linear = nn.Linear(config.MODEL.NUM_INPUT_FEATURES, config.SEGMENTATION.NUM_CLASSES - 1)
        if config.MODEL.COMPLETION_INTERACTION:
            self.shape_embedding = nn.Sequential(
                nn.Conv1d(config.MODEL.NUM_INPUT_FEATURES, config.MODEL.NUM_INPUT_FEATURES, kernel_size=1, bias=True),
                nn.InstanceNorm1d(config.MODEL.NUM_INPUT_FEATURES),
                nn.LeakyReLU(0.2,inplace=True),
            )
        self.criteria = F.cross_entropy


    def forward(self, coords, feat, label=None, weights=None):
        x = Me.SparseTensor(features=feat,
                            coordinates=coords.float(),
                            quantization_mode=Me.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        x = self.sparse_model(x)
        x = x.features_at_coordinates(coords.float()) # recover original shape
        if config.MODEL.COMPLETION_INTERACTION:
            feat = self.shape_embedding(x.unsqueeze(0).permute(0,2,1)).squeeze(0).permute(1,0)
            x = feat + x
            x = self.linear(x)
        else:
            feat = x
            x = self.linear(feat)

        loss = self.criteria(x, label.long(), weight=weights, reduction="mean", ignore_index=-100)
        return x, feat, loss
    
    def inference(self,coords,feat):
        x = Me.SparseTensor(features=feat,
                            coordinates=coords.float(),
                            quantization_mode=Me.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        x = self.sparse_model(x)
        x = x.features_at_coordinates(coords.float()) # recover original shape
        if config.MODEL.COMPLETION_INTERACTION:
            feat = self.shape_embedding(x.unsqueeze(0).permute(0,2,1)).squeeze(0).permute(1,0)
            x = feat + x
            x = self.linear(x)
        else:
            feat = x
            x = self.linear(feat)
        
        

        return x, feat