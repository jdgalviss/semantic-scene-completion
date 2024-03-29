import torch
from configs import config
import argparse
import numpy as np
from tqdm import tqdm
from model import MyModel
from utils import re_seed, get_test_dataloader
import os
import yaml
from pathlib import Path

device = torch.device("cuda:0")
shapes = {"256": torch.Size([1, 1, 256, 256, 32]), "128": torch.Size([1, 1, 128, 128, 16]), "64": torch.Size([1, 1, 64, 64, 8])}

def main():
    re_seed(0)
    # Test set dataloader
    test_dataloader = get_test_dataloader(config)

    # Model
    model = MyModel(is_teacher=False).to(device)
    # Load checkpoint
    ckpt_path = config.GENERAL.CHECKPOINT_PATH
    try:
        model.load_state_dict(torch.load(ckpt_path))
        training_epoch = int(ckpt_path.split('-')[-1].split('.')[0]) + 1
        print("TRAINING_EPOCH: ", training_epoch)
        print("Checkpoint {} found".format(ckpt_path))
    except:
        print("Checkpoint {} not found".format(ckpt_path))
        return

    model.eval()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: ", pytorch_total_params)
    print("Total trainable params: ", pytorch_total_trainable_params)
    print("\nEvaluating on {} samples".format(len(test_dataloader)))
    config.GENERAL.LEVEL = "FULL"
    config_file = os.path.join('configs/semantic-kitti.yaml')
    kitti_config = yaml.safe_load(open(config_file, 'r'))
    remapdict_inv = kitti_config["learning_map_inv"]

    with torch.no_grad():
        pbar = tqdm(test_dataloader)
        # Testing loop
        for i, batch in enumerate(pbar):
            filenames, complet_inputs, _, _ = batch
            # Forward pass
            try:
                results = model.inference(complet_inputs)
            except Exception as e:
                print(e, "Error in inference: ", i)
                del complet_inputs
                torch.cuda.empty_cache()
                continue
            occupancy_prediction = results['occupancy_{}'.format("256")]
            semantic_prediction = results['semantic_labels_{}'.format("256")]
            shape = shapes["256"]
            # transform predictions to dense tensors and remap
            min_coordinate = torch.IntTensor([0, 0, 0])
            occupancy_prediction, _, _ = occupancy_prediction.dense(shape, min_coordinate=min_coordinate)
            occupancy_prediction = occupancy_prediction.to("cpu")
            occupancy_prediction = occupancy_prediction > 0.5
            occupancy_prediction = np.uint16(occupancy_prediction[0,0].detach().cpu().numpy())
            semantic_prediction, _, _ = semantic_prediction.dense(shape, min_coordinate=min_coordinate)
            semantic_prediction = np.uint16(semantic_prediction.to("cpu")[0,0].detach().cpu().numpy())
            # level = "256"
            # print("semantic_prediction.shape: ", semantic_prediction.shape)
            # gt_labels = collect(complet_inputs, "complet_labels_{}".format(level)).squeeze()
            # gt_labels = np.uint16(gt_labels.detach().cpu().numpy())
            # semantic_prediction[gt_labels==255] = 0
            # print("gt.shape: ", semantic_prediction.shape)
            # remap
            maxkey = max(remapdict_inv.keys())
            remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
            remap_lut[list(remapdict_inv.keys())] = list(remapdict_inv.values())
            semantic_prediction = remap_lut[semantic_prediction]
            semantic_prediction = np.uint16(semantic_prediction)
            # save results
            try:
                directory = "output/test/sequences/{}/predictions".format(filenames[0][0])
                if not os.path.exists(directory):
                    print("creating folder: ", directory)
                    Path(directory).mkdir(parents=True, exist_ok=True)
            except:
                print("error saving files to dir: ",directory)
                continue
            filename = "output/test/sequences/{}/predictions/{}.label".format(filenames[0][0],filenames[0][1])
            semantic_prediction.astype('int16').tofile(filename)
            log_msg = {"sequence": filenames[0][0], "sample": filenames[0][1]}
            pbar.set_postfix(log_msg)
            torch.cuda.empty_cache()

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="Reinforcement Learning - based path planning")
    parser.add_argument("--config-file", type=str, default="configs/ssc_eval.yaml", required=False)
    parser.add_argument("--output-path", type=str, default="experiments", required=False)
    parser.add_argument("--checkpoint", type=str, default="/usr/stud/gaj/ssc/semantic-scene-completion/semantic-scene-completion/data/modelFULL-19.pth", required=False)

    args = parser.parse_args()
    config.merge_from_file(args.config_file)
    config.GENERAL.OUT_DIR = args.output_path
    config.GENERAL.CHECKPOINT_PATH = args.checkpoint
    print("\n Testing with configuration parameters: \n",config)
    main()