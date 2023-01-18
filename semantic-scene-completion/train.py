import torch
from configs import config
import argparse
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
import nvidia_smi


device = torch.device("cuda:0")

def main():
    re_seed(0)
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    train_dataset = SemanticKITTIDataset(config, "train",do_overfit=config.GENERAL.OVERFIT, num_samples_overfit=config.GENERAL.NUM_SAMPLES_OVERFIT, augment=config.TRAIN.AUGMENT)
    if config.GENERAL.OVERFIT:
        val_dataset = SemanticKITTIDataset(config, "train",do_overfit=True, num_samples_overfit=config.GENERAL.NUM_SAMPLES_OVERFIT)
        trainval_dataset = SemanticKITTIDataset(config, "train",do_overfit=True, num_samples_overfit=config.GENERAL.NUM_SAMPLES_OVERFIT)
    else:
        val_dataset = SemanticKITTIDataset(config, "valid",do_overfit=False)
        trainval_dataset = SemanticKITTIDataset(config, "trainval",do_overfit=False)

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

    trainval_dataloader = torch.utils.data.DataLoader(
        trainval_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        collate_fn=Merge,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        # worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )

    if config.GENERAL.OVERFIT:
        val_dataloaders = {"trainval": trainval_dataloader}
    else:
        val_dataloaders = {"val": val_dataloader, "trainval": trainval_dataloader}

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
    train_writer = SummaryWriter(log_dir=str(experiment_dir + "/train"))
    eval_writer = SummaryWriter(log_dir=str(experiment_dir + "/eval"))
    if config.GENERAL.OVERFIT:
        writers = {"trainval": train_writer}
    else:   
        writers = {"val": eval_writer, "trainval": train_writer}
    
    steps_schedule = config.TRAIN.STEPS
    iteration = training_epoch * len(train_dataloader)
    seg_label_to_cat = train_dataset.label_to_names
    model.train()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: ", pytorch_total_params)
    print("Total trainable params: ", pytorch_total_trainable_params)
    for epoch in range(training_epoch, training_epoch+int(config.TRAIN.MAX_EPOCHS)):
        # ============== Evaluation ==============
        if epoch % config.TRAIN.EVAL_PERIOD == 0 and config.GENERAL.LEVEL != "256":
            model.eval()
            with torch.no_grad():
                for dataloader_name, dataloader in val_dataloaders.items():
                    seg_evaluators = {"64": iouEval(config.SEGMENTATION.NUM_CLASSES, [0]),
                                      "128": iouEval(config.SEGMENTATION.NUM_CLASSES, [0]),  
                                      "256": iouEval(config.SEGMENTATION.NUM_CLASSES, [0]),}

                    occ_evaluators = {"64": iouEval(2, []),
                                        "128": iouEval(2, []),  
                                        "256": iouEval(2, []),}
                                    
                    print("\n------Evaluating {} set on {} samples------".format(dataloader_name, len(dataloader)))
                    if config.GENERAL.LEVEL == "64":
                        levels = ["64"]
                    elif config.GENERAL.LEVEL == "128":
                        levels = ["64", "128"]
                    elif config.GENERAL.LEVEL == "FULL":
                        levels = ["64", "128", "256"]
                    for i, batch in enumerate(tqdm(dataloader)):
                        _, complet_inputs, _, _ = batch
                        # try:
                        results = model.inference(complet_inputs)
                        # except Exception as e:
                            # print(e, "Error in inference: ", iteration)
                            # del complet_inputs
                            # torch.cuda.empty_cache()
                            # continue
                        # Semantic Eval

                        for level in levels:
                            semantic_gt = collect(complet_inputs,"complet_labels_{}".format(level)) #.long()
                            occupancy_gt = collect(complet_inputs,"complet_occupancy_{}".format(level)) #.long()
                            semantic_labels = results['semantic_labels_{}'.format(level)]

                            if level == "64":
                                occupancy_prediction = results['occupancy_{}'.format(level)].squeeze() > 0.5
                                occupancy_gt = occupancy_gt.to('cpu').data.numpy().flatten()
                                occupancy_prediction = occupancy_prediction.to('cpu').data.numpy().flatten()
                                occ_evaluators[level].addBatch(occupancy_prediction.astype(int), occupancy_gt.astype(int))

                                semantic_gt = semantic_gt.to('cpu').data.numpy().flatten()
                                semantic_labels = semantic_labels.to('cpu').data.numpy().flatten()
                                seg_labels_gt = semantic_gt[semantic_gt!=255]
                                semantic_labels = semantic_labels[semantic_gt!=255]
                                seg_evaluators[level].addBatch(semantic_labels.astype(int), seg_labels_gt.astype(int))
                            else:
                                occupancy_prediction = results['occupancy_{}'.format(level)]
                                occupancy_coordinates = occupancy_prediction.C.long()
                                occupancy_coordinates[:, 1:] = occupancy_coordinates[:, 1:] // occupancy_prediction.tensor_stride[0]
                                occupancy_gt = get_sparse_values(occupancy_gt.unsqueeze(0), occupancy_coordinates)
                                occupancy_prediction = occupancy_prediction.F > 0.5
                                occupancy_gt = occupancy_gt[:,0].to('cpu').data.numpy()
                                occupancy_prediction = occupancy_prediction[:,0].to('cpu').data.numpy()
                                occ_evaluators[level].addBatch(occupancy_prediction.astype(int), occupancy_gt.astype(int))

                                prediction = results['semantic_prediction_{}'.format(level)]
                                predicted_coordinates = prediction.C.long()
                                predicted_coordinates[:, 1:] = predicted_coordinates[:, 1:] // prediction.tensor_stride[0]
                                semantic_gt = get_sparse_values(semantic_gt.unsqueeze(0), predicted_coordinates)
                                semantic_labels = semantic_labels.F
                                torch.cuda.empty_cache()
                                semantic_gt = semantic_gt[:,0].to('cpu').data.numpy()
                                semantic_labels = semantic_labels[:,0].to('cpu').data.numpy()
                                seg_labels_gt = semantic_gt[semantic_gt!=255]
                                semantic_labels = semantic_labels[semantic_gt!=255]
                                seg_evaluators[level].addBatch(semantic_labels.astype(int), seg_labels_gt.astype(int))
                        # del complet_inputs, results, semantic_labels, semantic_gt, prediction, predicted_coordinates
                    for level in levels:
                        print("\nEvaluating level: {}".format(level))
                        occ_miou, occ_iou = occ_evaluators[level].getIoU()
                        m_jaccard, class_jaccard = seg_evaluators[level].getIoU()
                        ignore = []
                        for i, jacc in enumerate(class_jaccard):
                            if i not in ignore:
                                writers[dataloader_name].add_scalar('eval-{}/{}'.format(level,seg_label_to_cat[i]), jacc*100, epoch)
                                print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                                    i=i, class_str=seg_label_to_cat[i], jacc=jacc*100))
                        print('\n{} point avg class IoU: {}'.format(dataloader_name,m_jaccard*100))
                        print('{} point avg occupancy IoU: {}'.format(dataloader_name,occ_miou*100))
                        writers[dataloader_name].add_scalar('eval-{}/mIoU'.format(level), m_jaccard*100, epoch)
                        writers[dataloader_name].add_scalar('eval-{}/occ_mIoU'.format(level), occ_miou*100, epoch)
                    torch.cuda.empty_cache()

        # # ============== Training ==============
        if epoch>=steps_schedule[0]:
            if epoch>=steps_schedule[1]:
                if epoch>=steps_schedule[2]:
                    config.GENERAL.LEVEL = "FULL"
                else:
                    config.GENERAL.LEVEL = "256"
            else:
                config.GENERAL.LEVEL = "128"
        
        '''Adjust learning rate'''
        lr = max(config.SOLVER.BASE_LR * (config.SOLVER.LR_DECAY** (epoch // config.SOLVER.DECAY_STEP)), config.SOLVER.LR_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        train_writer.add_scalar('train/lr', lr, iteration)
    

        # with tqdm(total=len(train_dataloader)) as pbar:
        # for i, batch in enumerate(train_dataloader):
        pbar = tqdm(train_dataloader)
        consecutive_fails = 0
        val_dataloader_iterator = iter(val_dataloader)
        for i, batch in enumerate(pbar):
            model.train()
            optimizer.zero_grad()

            # print("batch len: ", len(batch))

            # Get tensors from batch
            _, complet_inputs, _, _ = batch
            # seg_labelweights = torch.Tensor(train_dataset.seg_labelweights).cuda()
            compl_labelweights = torch.Tensor(train_dataset.compl_labelweights).cuda()
            
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
                # Log memory
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                train_writer.add_scalar('memory/free', info.free, iteration)

                total_loss.backward()
                optimizer.step()
                log_msg = {"epoch": epoch}
                log_msg["level"] = config.GENERAL.LEVEL

                for k, v in losses.items():
                    # log_msg[k] = v.item()
                    # log_msg += "{}: {:.4f}, ".format(k, v)
                    if "256" in k:
                        train_writer.add_scalar('train_256/'+k, v.detach().cpu(), iteration)
                    elif "128" in k:
                        train_writer.add_scalar('train_128/'+k, v.detach().cpu(), iteration)
                    elif "64" in k:
                        train_writer.add_scalar('train_64/'+k, v.detach().cpu(), iteration)

                # log_msg += "total_loss: {:.4f}".format(total_loss)
                log_msg["total_loss"] = total_loss.item()
                # print(log_msg, end="\r")
                pbar.set_postfix(log_msg)

            train_writer.add_scalar('train/total_loss', total_loss.detach().cpu(), iteration)
            
            

            

            # Minkowski Engine recommendation
            torch.cuda.empty_cache()
            # del batch, complet_inputs

            # ============== Validation ==============
            model.eval()
            with torch.no_grad():
                try:
                    val_batch = next(val_dataloader_iterator)
                except StopIteration:
                    val_dataloader_iterator = iter(val_dataloader)
                    val_batch = next(val_dataloader_iterator)
                # Get tensors from batch
                _, complet_inputs, _, _ = val_batch
                # seg_labelweights = torch.Tensor(train_dataset.seg_labelweights).cuda()
                compl_labelweights = torch.Tensor(train_dataset.compl_labelweights).cuda()
                
                # Forward pass through model
                # try:
                losses, _ = model(complet_inputs, compl_labelweights)
                # except Exception as e:
                #     print(e, "Error in val forward pass: ", iteration)
                #     del complet_inputs
                #     torch.cuda.empty_cache()
                #     continue
                total_loss: torch.Tensor = 0.0
                total_loss = losses["occupancy_64"] * config.MODEL.OCCUPANCY_64_WEIGHT + losses["semantic_64"]*config.MODEL.SEMANTIC_64_WEIGHT 
                if config.GENERAL.LEVEL == "128" or config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                    total_loss += losses["occupancy_128"] * config.MODEL.OCCUPANCY_128_WEIGHT + losses["semantic_128"]*config.MODEL.SEMANTIC_128_WEIGHT 
                if config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                    total_loss += losses["occupancy_256"]*config.MODEL.OCCUPANCY_256_WEIGHT 
                if config.GENERAL.LEVEL == "FULL":
                    total_loss += losses["semantic_256"]*config.MODEL.SEMANTIC_256_WEIGHT
                if torch.is_tensor(total_loss):
                    for k, v in losses.items():
                        # log_msg[k] = v.item()
                        # log_msg += "{}: {:.4f}, ".format(k, v)
                        if "256" in k:
                            eval_writer.add_scalar('train_256/'+k, v.detach().cpu(), iteration)
                        elif "128" in k:
                            eval_writer.add_scalar('train_128/'+k, v.detach().cpu(), iteration)
                        elif "64" in k:
                            eval_writer.add_scalar('train_64/'+k, v.detach().cpu(), iteration)

                eval_writer.add_scalar('train/total_loss', total_loss.detach().cpu(), iteration)
            
            # Log memory
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            eval_writer.add_scalar('memory/free', info.free, iteration)

            iteration += 1
            torch.cuda.empty_cache()
            # ========================================
        eval_writer.add_scalar('epoch', epoch, iteration)

        # Save checkpoint
        if epoch % config.TRAIN.CHECKPOINT_PERIOD == 0:
            torch.save(model.state_dict(), experiment_dir + "/model{}-{}.pth".format(config.GENERAL.LEVEL, epoch))

            
    torch.save(model.state_dict(), experiment_dir + "/model.pth")



if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="Semantic Scene Completion")
    parser.add_argument("--config-file", type=str, default="configs/ssc.yaml", required=False)
    parser.add_argument("--output-path", type=str, default="experiments", required=False)

    args = parser.parse_args()
    config.GENERAL.OUT_DIR = args.output_path
    config.merge_from_file(args.config_file)
    print("\n Training with configuration parameters: \n",config)

    main()

