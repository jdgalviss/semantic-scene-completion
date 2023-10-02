import sys
sys.path.append("/usr/stud/gaj/ssc/semantic-scene-completion/semantic-scene-completion/thirdparty/Codes-for-SCPNet")
from dataloader.pc_dataset import get_SemKITTI_label_name, get_eval_mask, unpack
from builder import data_builder, model_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint
import os
import torch.nn as nn
import torch
pytorch_device = torch.device('cuda:0')


class SCPNetPriorsModel(nn.Module):
    def __init__(self):
        super().__init__()
        config_path = '/usr/stud/gaj/ssc/semantic-scene-completion/semantic-scene-completion/thirdparty/Codes-for-SCPNet/config/semantickitti-multiscan2.yaml'
        configs = load_config_data(config_path)
        model_config = configs['model_params']
        train_hypers = configs['train_params']
        self.scpnet_model = model_builder.build(model_config)
        model_load_path = train_hypers['model_load_path']
        model_load_path += 'scpnet_iou37.5557_epoch3.pth'
        # model_load_path += '0.pth'
        # model_save_path += ''

        if os.path.exists(model_load_path):
            print('Load model from: %s' % model_load_path)
            self.scpnet_model = load_checkpoint(model_load_path, self.scpnet_model)
        else:
            print('No existing model, training model from scratch...')

        # Clear seg sugnetwork from scpnet
        del self.scpnet_model.cylinder_3d_spconv_seg.downCntx
        del self.scpnet_model.cylinder_3d_spconv_seg.resBlock2
        del self.scpnet_model.cylinder_3d_spconv_seg.resBlock3
        del self.scpnet_model.cylinder_3d_spconv_seg.resBlock4
        del self.scpnet_model.cylinder_3d_spconv_seg.resBlock5
        del self.scpnet_model.cylinder_3d_spconv_seg.upBlock0
        del self.scpnet_model.cylinder_3d_spconv_seg.upBlock1
        del self.scpnet_model.cylinder_3d_spconv_seg.upBlock2
        del self.scpnet_model.cylinder_3d_spconv_seg.upBlock3
        del self.scpnet_model.cylinder_3d_spconv_seg.ReconNet
        del self.scpnet_model.cylinder_3d_spconv_seg.logits
        torch.cuda.empty_cache()
        pytorch_total_trainable_params = sum(p.numel() for p in self.scpnet_model.parameters() if p.requires_grad)
        print("Total trainable params: ", pytorch_total_trainable_params)


    def forward(self,train_pt_fea_ten, train_vox_ten, batch_size=1):
        coords, features = self.scpnet_model(train_pt_fea_ten, train_vox_ten, batch_size)

        return coords, features
