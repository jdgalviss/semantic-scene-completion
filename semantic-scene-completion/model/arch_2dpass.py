import torch
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet34
from .lovasz_loss import Lovasz_loss
import torch_scatter
from structures import collect

import numpy as np
class ResNetFCN(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, config=None):
        super(ResNetFCN, self).__init__()

        if backbone == "resnet34":
            net = resnet34(pretrained)
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))
        self.hiden_size = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(128, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )

    def forward(self, image, img_indices):
        x = image
        x = x.permute(0,3,1,2)
        h, w = x.shape[2], x.shape[3]
        if h % 16 != 0 or w % 16 != 0:
            assert False, "invalid input size: {}".format(x.shape)

        # Encoder
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(self.maxpool(conv1_out))
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        # Deconv
        layer1_out = self.deconv_layer1(layer1_out)
        layer2_out = self.deconv_layer2(layer2_out)
        layer3_out = self.deconv_layer3(layer3_out)
        layer4_out = self.deconv_layer4(layer4_out)

        result = {}

        result['img_scale2'] = layer1_out # All feature maps have size 1,64,320,480
        result['img_scale4'] = layer2_out
        result['img_scale8'] = layer3_out
        result['img_scale16'] = layer4_out
        print("layer1_out.shape: ", layer1_out.shape)
        print("layer2_out.shape: ", layer2_out.shape)
        print("layer3_out.shape: ", layer3_out.shape)
        print("layer4_out.shape: ", layer4_out.shape)

        process_keys = [k for k in result.keys() if k.find('img_scale') != -1]

        temp = {k: [] for k in process_keys}

        for i in range(x.shape[0]):
            for k in process_keys:
                temp[k].append(result[k].permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])

        for k in process_keys:
            result[k] = torch.cat(temp[k], 0)

        return result
    
class xModalKD(nn.Module):
    def __init__(self, config=None):
        super(xModalKD, self).__init__()
        self.hiden_size = 64
        self.scale_list = [2,4,8,16]
        self.num_classes = 20
        self.lambda_xm = 0.05
        self.lambda_seg2d = 1
        self.num_scales = len(self.scale_list)
        self.sizes_per_scale = [128,64,64,64]

        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )

        self.multihead_fuse_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_fuse_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
        self.leaners = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        for i in range(self.num_scales):
            self.leaners.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
            self.fcs1.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))
            self.fcs2.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))

        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        # if 'seg_labelweights' in config['dataset_params']:
        #     seg_num_per_class = config['dataset_params']['seg_labelweights']
        #     seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
        #     seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        # else:
        seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(weight=seg_labelweights, ignore_index=255)
        self.lovasz_loss = Lovasz_loss(ignore=255)

    @staticmethod
    def p2img_mapping(pts_fea, p2img_idx, batch_idx):
        img_feat = []
        for b in range(batch_idx.max()+1):
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
        return torch.cat(img_feat, 0)

    @staticmethod
    def voxelize_labels(labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
        labels = unq_lbxyz[:, 0][label_ind]
        return labels

    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss

    def fusion_to_single_KD(self, image_feat, feature_completion, label):
        # batch_idx = data_dict['batch_idx']
        # point2img_index = data_dict['point2img_index']
        # last_scale = self.scale_list[idx - 1] if idx > 0 else 1
        # img_feat = data_dict['img_scale{}'.format(self.scale_list[idx])]
        # pts_feat = data_dict['layer_{}'.format(idx)]['pts_feat']
        # coors_inv = data_dict['scale_{}'.format(last_scale)]['coors_inv']
        return 0,0
        # 3D prediction
        pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)

        # correspondence
        pts_label_full = self.voxelize_labels(data_dict['labels'], data_dict['layer_{}'.format(idx)]['full_coors'])
        pts_feat = self.p2img_mapping(pts_feat[coors_inv], point2img_index, batch_idx)
        pts_pred = self.p2img_mapping(pts_pred_full[coors_inv], point2img_index, batch_idx)

        # modality fusion
        feat_learner = F.relu(self.leaners[idx](pts_feat))
        feat_cat = torch.cat([img_feat, feat_learner], 1)
        feat_cat = self.fcs1[idx](feat_cat)
        feat_weight = torch.sigmoid(self.fcs2[idx](feat_cat))
        fuse_feat = F.relu(feat_cat * feat_weight)

        # fusion prediction
        fuse_pred = self.multihead_fuse_classifier[idx](fuse_feat)

        # Segmentation Loss
        seg_loss_3d = self.seg_loss(pts_pred_full, pts_label_full)
        seg_loss_2d = self.seg_loss(fuse_pred, data_dict['img_label'])
        loss = seg_loss_3d + seg_loss_2d * self.lambda_seg2d / self.num_scales

        # KL divergence
        xm_loss = F.kl_div(
            F.log_softmax(pts_pred, dim=1),
            F.softmax(fuse_pred.detach(), dim=1),
        )
        loss += xm_loss * self.lambda_xm / self.num_scales

        return loss, fuse_feat

    def forward(self, image_feats, features_completion, targets):
        loss = 0
        img_seg_feat = []
        for idx in range(1): #range(self.num_scales):
            label = collect(targets, "complet_labels_{}".format(self.sizes_per_scale[idx]))
            if idx==2:
                label = F.max_pool3d(label.float(), kernel_size=2, stride=2)
            if idx == 3:
                label = F.max_pool3d(label.float(), kernel_size=4, stride=4)
            print("label: ", label.shape)
            singlescale_loss, fuse_feat = self.fusion_to_single_KD(image_feats["img_scale{}".format(self.scale_list[idx])], 
                                                                   features_completion[idx], label)
            # img_seg_feat.append(fuse_feat)
            # loss += singlescale_loss
        return 0
        img_seg_logits = self.classifier(torch.cat(img_seg_feat, 1))
        loss += self.seg_loss(img_seg_logits, data_dict['img_label'])
        data_dict['loss'] += loss

        return data_dict