import torch.optim as optim
import torch
from configs import config
import argparse
import shutil
from semantic_kitti_dataset import SemanticKITTIDataset, Merge
import numpy as np
import time
from tqdm import tqdm
from model import MyModel
import MinkowskiEngine as Me
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.path_utils import create_new_experiment_folder, save_config
from utils import re_seed
from structures import collect


device = torch.device("cuda:0")


def main():
    re_seed(123)
    train_dataset = SemanticKITTIDataset(config, "train")
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        collate_fn=Merge,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )
    model = MyModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), config.SOLVER.BASE_LR,
                                          betas=(config.SOLVER.BETA_1, config.SOLVER.BETA_2),
                                          weight_decay=config.SOLVER.WEIGHT_DECAY)

    experiment_dir = create_new_experiment_folder(config.GENERAL.OUT_DIR)
    save_config(config, experiment_dir)
    writer = SummaryWriter(log_dir=str(experiment_dir + "/tensorboard"))

    training_epoch = 0
    steps_schedule = [10,20]
    iteration = 0
    for epoch in range(training_epoch, training_epoch+100):
        model.train()
        if epoch>=steps_schedule[0]:
            if epoch>=steps_schedule[1]:
                config.GENERAL.LEVEL = "FULL"
            else:
                config.GENERAL.LEVEL = "256"
        # with tqdm(total=len(train_data_loader)) as pbar:
        for i, batch in enumerate(train_data_loader):
            optimizer.zero_grad()

            # print("batch len: ", len(batch))

            # Get tensors from batch
            _, complet_inputs, _, _ = batch
            # complet_coords = complet_inputs['complet_coords'].to(device)
            # complet_invalid = complet_inputs['complet_invalid'].to(device)
            # complet_labels = complet_inputs['complet_labels'].to(device)
            complet_coords = collect(complet_inputs, "complet_coords").squeeze()
            complet_labels = collect(complet_inputs, "complet_labels")
            complet_invalid = collect(complet_inputs, "complet_invalid")

            # Forward pass through model
            losses, _ = model(complet_coords, complet_invalid, complet_labels)
            
            total_loss = losses["occupancy_128"] * config.MODEL.OCCUPANCY_128_WEIGHT + losses["semantic_128"]*config.MODEL.SEMANTIC_128_WEIGHT 
            if config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                total_loss += losses["occupancy_256"]*config.MODEL.OCCUPANCY_256_WEIGHT 
            if config.GENERAL.LEVEL == "FULL":
                total_loss += losses["semantic_256"]*config.MODEL.SEMANTIC_256_WEIGHT
            # Loss backpropagation, optimizer & scheduler step

            if torch.is_tensor(total_loss):
                total_loss.backward()
                optimizer.step()
                log_msg = "\r step: {}/{}, ".format(epoch, i)
                for k, v in losses.items():
                    log_msg += "{}: {:.4f}, ".format(k, v)
                    if "256" in k:
                        writer.add_scalar('train_256/'+k, v.detach().cpu(), iteration)
                    elif "128" in k:
                        writer.add_scalar('train_128/'+k, v.detach().cpu(), iteration)

                log_msg += "total_loss: {:.4f}".format(total_loss)
                print(log_msg)
            writer.add_scalar('train/total_loss', total_loss.detach().cpu(), iteration)
            iteration += 1
                
            # Minkowski Engine recommendation
            torch.cuda.empty_cache()


    torch.save(model.state_dict(), experiment_dir + "/model.pth")


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="Reinforcement Learning - based path planning")
    parser.add_argument("--config-file", type=str, default="configs/ssc.yaml", required=False)
    parser.add_argument("--output-path", type=str, default="experiments", required=False)

    args = parser.parse_args()
    config.GENERAL.OUT_DIR = args.output_path
    print("\n Training with configuration parameters: \n",config)
    config.merge_from_file(args.config_file)

    main()

