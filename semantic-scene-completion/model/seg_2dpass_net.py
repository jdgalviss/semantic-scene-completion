from easydict import EasyDict
import torch
import argparse
import yaml
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.append("/usr/src/app/semantic-scene-completion/thirdparty/2DPASS")
from network.spvcnn import get_model as SPVCNN
import os
import importlib
from configs import config




device = torch.device("cuda:0")

class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print(f'Error: {message}')

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config

def parse_config():
    parser = CustomArgumentParser(description='Your script description')
    # general
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,), help='specify gpu devices')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--config_path', default='thirdparty/2DPASS/config/2DPASS-semantickitti.yaml')
    # training
    parser.add_argument('--log_dir', type=str, default='default', help='log location')
    parser.add_argument('--monitor', type=str, default='val/mIoU', help='the maximum metric')
    parser.add_argument('--stop_patience', type=int, default=50, help='patience for stop training')
    parser.add_argument('--save_top_k', type=int, default=1, help='save top k checkpoints, use -1 to checkpoint every epoch')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')
    parser.add_argument('--SWA', action='store_true', default=False, help='StochasticWeightAveraging')
    parser.add_argument('--baseline_only', action='store_true', default=False, help='training without 2D')
    # testing
    parser.add_argument('--test', action='store_true', default=True, help='test mode')
    parser.add_argument('--fine_tune', action='store_true', default=False, help='fine tune mode')
    parser.add_argument('--pretrain2d', action='store_true', default=False, help='use pre-trained 2d network')
    parser.add_argument('--num_vote', type=int, default=1, help='number of voting in the test')
    parser.add_argument('--submit_to_server', action='store_true', default=False, help='submit on benchmark')
    parser.add_argument('--checkpoint', type=str, default="thirdparty/2DPASS/best_model.ckpt", help='load checkpoint')
    # debug
    parser.add_argument('--debug', default=False, action='store_true')

    args, unknown = parser.parse_known_args()
    config = load_yaml(args.config_path)
    config.update(vars(args))  # override the configuration using the value in args

    # voting test
    if args.test:
        config['dataset_params']['val_data_loader']['batch_size'] = args.num_vote
    if args.num_vote > 1:
        config['dataset_params']['val_data_loader']['rotate_aug'] = True
        config['dataset_params']['val_data_loader']['transform_aug'] = True
    if args.debug:
        config['dataset_params']['val_data_loader']['batch_size'] = 2
        config['dataset_params']['val_data_loader']['num_workers'] = 0

    return EasyDict(config)

class SparseSegNet2DPASS(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        configs = parse_config()
        print("2DPASS config: ", configs)
        self.model = SPVCNN(configs)
        # Load model checkpoint
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, configs.gpu))
        num_gpu = len(configs.gpu)
        model_file = importlib.import_module('network.' + configs['model_params']['model_architecture'])
        my_model = model_file.get_model(configs)

        if configs.checkpoint is not None:
            print('loading pre-trained segmentation model...')
            my_model = my_model.load_from_checkpoint(configs.checkpoint, config=configs, strict=(not configs.pretrain2d))

        sd=my_model.model_3d.state_dict()
        self.model.load_state_dict(sd)
        del my_model
        self.model = self.model.to(device)

    def forward(self, coords, feat, label=None, weights=None):
        data_dict = {}
        data_dict['points'] = feat #xyz + feat
        data_dict['batch_idx'] = coords[:,0]
        data_dict['batch_size'] = torch.max(coords[:,0]) + 1 # TODO: enable batch size > 1
        data_dict['labels'] = label
        result_dict = self.model(data_dict)
        out = result_dict['logits'][:,1:]
        if config.SEGMENTATION.SOFTMAX:
            out = F.softmax(out, dim=1)

        return out, result_dict['features'], result_dict['loss']
    
    def inference(self, coords, feat):
        data_dict = {}
        data_dict['points'] = feat
        data_dict['batch_idx'] = coords[:,0]
        data_dict['batch_size'] = torch.max(coords[:,0]) + 1 # TODO: enable batch size > 1
        data_dict['labels'] = None
        result_dict = self.model(data_dict)
        out = result_dict['logits'][:,1:]
        if config.SEGMENTATION.SOFTMAX:
            out = F.softmax(out, dim=1)
        return out, result_dict['features']



