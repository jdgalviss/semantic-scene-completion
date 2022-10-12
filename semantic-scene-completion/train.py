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



device = torch.device("cuda:0")



def main():
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

    training_epoch = 0
    steps_schedule = [10,20]
    for epoch in range(training_epoch, training_epoch+200):
        model.train()
        if epoch>steps_schedule[0]:
            if epoch>steps_schedule[1]:
                config.GENERAL.LEVEL = "FULL"
            else:
                config.GENERAL.LEVEL = "256"
        # with tqdm(total=len(train_data_loader)) as pbar:
        for i, batch in enumerate(train_data_loader):
            optimizer.zero_grad()

            # print("batch len: ", len(batch))

            # Get tensors from batch
            _, complet_inputs, completion_collection, _ = batch
            complet_coords = complet_inputs['complet_coords'].to(device)
            complet_invalid = complet_inputs['complet_invalid'].to(device)
            complet_labels = complet_inputs['complet_labels'].to(device)

            # Forward pass through model
            losses = model(complet_coords, complet_invalid, complet_labels)
            
            total_loss = losses['128/occupancy'] * config.MODEL.OCCUPANCY_128_WEIGHT + losses["128/semantic"]*config.MODEL.SEMANTIC_128_WEIGHT 
            if config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                total_loss += losses["256/occupancy"]*config.MODEL.OCCUPANCY_256_WEIGHT 
            if config.GENERAL.LEVEL == "FULL":
                total_loss += losses["256/semantic"]*config.MODEL.SEMANTIC_256_WEIGHT
            # Loss backpropagation, optimizer & scheduler step

            if torch.is_tensor(total_loss):
                total_loss.backward()
                optimizer.step()
                log_msg = "\r step: {}, ".format(epoch)
                for k, v in losses.items():
                    log_msg += "{}: {:.4f}, ".format(k, v)
                log_msg += "total_loss: {:.4f}".format(total_loss)
                print(log_msg)
                
            # Minkowski Engine recommendation
            torch.cuda.empty_cache()

            
            # Occupancy predictions
            
            continue
            output_features = predictions[0]
            print("output_features: ", output_features.shape)
            # occupancy_prediction_128 = predictions[1][0]
            # print("occupancy_prediction_128: ", occupancy_prediction_128.shape)
            occupancy_prediction = occupancy_256_head(output_features)
            print("occupancy_prediction: ",occupancy_prediction.shape)
            occupancy_prediction = mask_invalid_sparse_voxels(occupancy_prediction)
            predicted_coordinates = occupancy_prediction.C.long()
            predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // occupancy_prediction.tensor_stride[0]
            print("occupancy_prediction: ",occupancy_prediction.shape)
            occupancy_prediction = Me.MinkowskiSigmoid()(occupancy_prediction)
            print("occupancy_prediction: ",occupancy_prediction.shape)
            dense_dimensions = torch.Size([1, 1] + [256, 256, 32])
            min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
            occupancy_prediction_dense, _, _ = occupancy_prediction.dense(dense_dimensions, min_coordinates)
            print("Occupancy prediction dense: ", occupancy_prediction_dense.shape)

            # Use occupancy prediction to refine sparse voxels
            sparse_threshold_256 = 0.5
            occupancy_mask = (occupancy_prediction.F > sparse_threshold_256).squeeze()
            print("occupancy_mask: ", occupancy_mask.shape)
            print("occupancy_mask values: ",torch.unique(occupancy_mask))
            output_features = Me.MinkowskiPruning()(output_features, occupancy_mask)

            # Semantic predictions
            semantic_prediction = semantic_head(output_features)
            print("semantic_prediction: ", semantic_prediction.shape)
            print("semantic_prediction values: ", torch.unique(semantic_prediction))


            # print("complet_labels: ",complet_labels.shape)

            

            # shape = torch.Size([1,1,128, 128, 16])
            # min_coordinate = torch.IntTensor([0, 0, 0]).to(device)
            # occupancy_prediction_128_dense, _, _ = occupancy_prediction_128.dense(shape, min_coordinate=min_coordinate)
            # print("occupancy_prediction_128_dense: ", occupancy_prediction_128_dense.shape)
            ## Process seg_inputs with completion head
                


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="Reinforcement Learning - based path planning")
    parser.add_argument("--config-file", type=str, default="configs/ssc.yaml", required=False)
    parser.add_argument("--output-path", type=str, default="experiments", required=False)

    args = parser.parse_args()
    # config.merge_from_file(args.config_file)

    print("\n Training with configuration parameters: {}\n".format(config))

    main()