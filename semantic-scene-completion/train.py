import torch.optim as optim
import torch
from configs import config
import argparse
import shutil
from semantic_kitti_dataset import SemanticKITTIDataset
import numpy as np
import time
from tqdm import tqdm

def main():
    train_dataset = SemanticKITTIDataset(config, "train")
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )
    training_epoch = 0
    for epoch in range(training_epoch, training_epoch+1):
        with tqdm(total=len(train_data_loader)) as pbar:
            for i, batch in enumerate(train_data_loader):
                print("batch keys: ", batch.keys())
                # seg_label = batch[0]['seg_labels']
                # complet_label = batch[1]['complet_labels']
                # invalid_voxels = batch[1]['complet_invalid']

                # print("seg_label", seg_label.shape)
                # print("complet_label", complet_label.shape)
                # print("invalid_voxels", invalid_voxels.shape)

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="Reinforcement Learning - based path planning")
    parser.add_argument("--config-file", type=str, default="configs/ssc.yaml", required=False)
    parser.add_argument("--output-path", type=str, default="experiments", required=False)

    args = parser.parse_args()
    # config.merge_from_file(args.config_file)

    print("\n Training with configuration parameters: {}\n".format(config))

    main()