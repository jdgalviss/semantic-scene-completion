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
from evaluation import iouEval
from model import get_sparse_values

device = torch.device("cuda:0")


def main():
    re_seed(0)
    train_dataset = SemanticKITTIDataset(config, "train",do_overfit=config.GENERAL.OVERFIT, num_samples_overfit=config.GENERAL.NUM_SAMPLES_OVERFIT)
    if config.GENERAL.OVERFIT:
        val_dataset = SemanticKITTIDataset(config, "train",do_overfit=True, num_samples_overfit=config.GENERAL.NUM_SAMPLES_OVERFIT)
    else:
        val_dataset = SemanticKITTIDataset(config, "valid",do_overfit=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        collate_fn=Merge,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        collate_fn=Merge,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        # worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )

    model = MyModel(num_output_channels=config.MODEL.NUM_OUTPUT_CHANNELS, unet_features=config.MODEL.UNET_FEATURES).to(device)
    if config.GENERAL.CHECKPOINT_PATH is not None:
        model.load_state_dict(torch.load(config.GENERAL.CHECKPOINT_PATH))
        training_epoch = int(config.GENERAL.CHECKPOINT_PATH.split('-')[-1].split('.')[0]) + 1
        print("TRAINING_EPOCH: ", training_epoch)
    else:
        training_epoch = 0
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), config.SOLVER.BASE_LR,
                                          betas=(config.SOLVER.BETA_1, config.SOLVER.BETA_2),
                                          weight_decay=config.SOLVER.WEIGHT_DECAY)

    experiment_dir = create_new_experiment_folder(config.GENERAL.OUT_DIR)
    save_config(config, experiment_dir)
    writer = SummaryWriter(log_dir=str(experiment_dir + "/tensorboard"))

    
    steps_schedule = config.TRAIN.STEPS
    iteration = training_epoch * len(train_dataloader)
    seg_label_to_cat = train_dataset.label_to_names
    model.train()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: ", pytorch_total_params)
    print("Total trainable params: ", pytorch_total_trainable_params)
    for epoch in range(training_epoch, training_epoch+int(config.TRAIN.MAX_EPOCHS)):
        model.train()
        if epoch>=steps_schedule[0]:
            if epoch>=steps_schedule[1]:
                if epoch>=steps_schedule[2]:
                    config.GENERAL.LEVEL = "FULL"
                else:
                    config.GENERAL.LEVEL = "256"
            else:
                config.GENERAL.LEVEL = "128"
        # with tqdm(total=len(train_dataloader)) as pbar:
        # for i, batch in enumerate(train_dataloader):
        pbar = tqdm(train_dataloader)
        consecutive_fails = 0
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()

            # print("batch len: ", len(batch))

            # Get tensors from batch
            _, complet_inputs, _, _ = batch
            # seg_labelweights = torch.Tensor(train_dataset.seg_labelweights).cuda()
            compl_labelweights = torch.Tensor(train_dataset.compl_labelweights).cuda()
            # print("seg_labelweights: ", seg_labelweights.shape)
            # print("seg_labelweights: ", torch.unique(seg_labelweights))
            # print("compl_labelweights: ", compl_labelweights.shape)
            # print("compl_labelweights: ", (compl_labelweights))
            # complet_coords = complet_inputs['complet_coords'].to(device)
            # complet_invalid = complet_inputs['complet_invalid'].to(device)
            # complet_labels = complet_inputs['complet_labels'].to(device)
            # complet_coords = collect(complet_inputs, "complet_coords").squeeze()
            # complet_labels = collect(complet_inputs, "complet_labels")
            # complet_invalid = collect(complet_inputs, "complet_invalid")

            # Forward pass through model
            try:
                losses, _ = model(complet_inputs, compl_labelweights)
            except Exception as e:
                print(e, "Error in forward pass: ", iteration)
                consecutive_fails += 1
                if consecutive_fails > 20:
                    print("Too many consecutive fails, exiting")
                    return
                del complet_inputs
                torch.cuda.empty_cache()
                continue
            consecutive_fails = 0
            total_loss: torch.Tensor = 0.0
            total_loss = losses["occupancy_64"] * config.MODEL.OCCUPANCY_64_WEIGHT + losses["semantic_64"]*config.MODEL.SEMANTIC_64_WEIGHT 
            if config.GENERAL.LEVEL == "128" or config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                total_loss += losses["occupancy_128"] * config.MODEL.OCCUPANCY_128_WEIGHT + losses["semantic_128"]*config.MODEL.SEMANTIC_128_WEIGHT 
            if config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                total_loss += losses["occupancy_256"]*config.MODEL.OCCUPANCY_256_WEIGHT 
            if config.GENERAL.LEVEL == "FULL":
                total_loss += losses["semantic_256"]*config.MODEL.SEMANTIC_256_WEIGHT
            # Loss backpropagation, optimizer & scheduler step

            if torch.is_tensor(total_loss):
                total_loss.backward()
                optimizer.step()
                log_msg = {"epoch": epoch}
                log_msg["level"] = config.GENERAL.LEVEL

                for k, v in losses.items():
                    # log_msg[k] = v.item()
                    # log_msg += "{}: {:.4f}, ".format(k, v)
                    if "256" in k:
                        writer.add_scalar('train_256/'+k, v.detach().cpu(), iteration)
                    elif "128" in k:
                        writer.add_scalar('train_128/'+k, v.detach().cpu(), iteration)
                    elif "64" in k:
                        writer.add_scalar('train_64/'+k, v.detach().cpu(), iteration)

                # log_msg += "total_loss: {:.4f}".format(total_loss)
                log_msg["total_loss"] = total_loss.item()
                # print(log_msg, end="\r")
                pbar.set_postfix(log_msg)

            writer.add_scalar('train/total_loss', total_loss.detach().cpu(), iteration)
            iteration += 1
                
            # Minkowski Engine recommendation
            torch.cuda.empty_cache()
            writer.add_scalar('epoch', epoch, iteration)

        # Save checkpoint
        if epoch % config.TRAIN.CHECKPOINT_PERIOD == 0:
            torch.save(model.state_dict(), experiment_dir + "/model{}-{}.pth".format(config.GENERAL.LEVEL, epoch))

        # # ============== Evaluation ==============
        if epoch % config.TRAIN.EVAL_PERIOD == 0 and config.GENERAL.LEVEL == "FULL":
            model.eval()
            print("\nEvaluating on {} samples".format(len(val_dataloader)))
            with torch.no_grad():
                seg_evaluator = iouEval(config.SEGMENTATION.NUM_CLASSES, [0,20,21])

                # for i, batch in enumerate(val_dataloader):
                for i, batch in enumerate(tqdm(val_dataloader)):
                    _, complet_inputs, _, _ = batch
                    
                    try:
                        results = model.inference(complet_inputs)
                    except Exception as e:
                        print(e, "Error in inference: ", iteration)
                        del complet_inputs
                        torch.cuda.empty_cache()
                        continue
                    # Semantic Eval
                    semantic_prediction = results['semantic_256']
                    prediction = results['semantic_prediction']
                    semantic_labels = collect(complet_inputs,'complet_labels') #.long()
                    predicted_coordinates = prediction.C.long()
                    predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]
                    semantic_labels = get_sparse_values(semantic_labels.unsqueeze(0), predicted_coordinates)
                    semantic_prediction = semantic_prediction.F
                    torch.cuda.empty_cache()
                    semantic_labels = semantic_labels[:,0].to('cpu').data.numpy()
                    semantic_prediction = semantic_prediction[:,0].to('cpu').data.numpy()
                    seg_labels = semantic_labels[semantic_labels!=255]
                    semantic_prediction = semantic_prediction[semantic_labels!=255]
                    seg_evaluator.addBatch(semantic_prediction.astype(int), seg_labels.astype(int))
                    del complet_inputs, results, semantic_prediction, semantic_labels, prediction, predicted_coordinates

                _, class_jaccard = seg_evaluator.getIoU()
                m_jaccard = class_jaccard.mean()
                print("mean_iou: ", m_jaccard)
                ignore = [0]
                for i, jacc in enumerate(class_jaccard):
                    if i not in ignore:
                        writer.add_scalar('eval/{}'.format(seg_label_to_cat[i]), jacc*100, iteration)
                        print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                            i=i, class_str=seg_label_to_cat[i], jacc=jacc*100))
                print('\nEval point avg class IoU: %f \n' % (m_jaccard*100))
                writer.add_scalar('eval/mIoU', m_jaccard*100, iteration)
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

