import torch
from configs import config, LABEL_TO_NAMES
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import nvidia_smi
import time
from tqdm import tqdm
from torch.nn import functional as F
import torchvision


from model import MyModel
from structures import collect
from semantic_kitti_dataset import get_labelweights
from utils import re_seed, labels_to_cmap2d, get_bev, input_to_cmap2d, get_dataloaders, update_level
from utils.path_utils import create_new_experiment_folder, save_config
from evaluation import iouEval



epsilon = np.finfo(np.float32).eps
device = torch.device("cuda:0")
eval_imgs_idxs = [1,2,3,4,5,6,7,8]

def main():
    re_seed(0)
    # measuring gpu memory
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # completion different voxel tensors sizes
    shapes = {"256": torch.Size([1, 1, 256, 256, 32]), "128": torch.Size([1, 1, 128, 128, 16]), "64": torch.Size([1, 1, 64, 64, 8])}

    # Get dataloaders
    train_dataloader, val_dataloader, trainval_dataloader = get_dataloaders(config) 
    if config.GENERAL.OVERFIT:
        val_dataloaders = {"trainval": trainval_dataloader}
    else:
        val_dataloaders = {"val": val_dataloader, "trainval": trainval_dataloader}
    
    # SSC Model
    model = MyModel().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), config.SOLVER.BASE_LR,
                                          betas=(config.SOLVER.BETA_1, config.SOLVER.BETA_2),
                                          weight_decay=config.SOLVER.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.SOLVER.LR_DECAY_RATE)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*300, eta_min=config.SOLVER.LR_CLIP)

    # Load checkpoint
    if config.GENERAL.CHECKPOINT_PATH is not None:
        model.load_state_dict(torch.load(config.GENERAL.CHECKPOINT_PATH))
        training_epoch = int(config.GENERAL.CHECKPOINT_PATH.split('-')[-1].split('.')[0]) + 1
        print("TRAINING_EPOCH: ", training_epoch)
    else:
        training_epoch = 0
    
    iteration = training_epoch * len(train_dataloader)
    seg_label_to_cat = LABEL_TO_NAMES #TODO: replace

    # Number of Parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: ", pytorch_total_params)
    print("Total trainable params: ", pytorch_total_trainable_params)

    # Setup logging
    experiment_dir = create_new_experiment_folder(config.GENERAL.OUT_DIR)
    save_config(config, experiment_dir)
    train_writer = SummaryWriter(log_dir=str(experiment_dir + "/train"))
    if config.GENERAL.OVERFIT:
        writers = {"trainval": train_writer}
    else:   
        eval_writer = SummaryWriter(log_dir=str(experiment_dir + "/eval"))
        writers = {"val": eval_writer, "trainval": train_writer}
    
    # get labelweights
    seg_labelweights, compl_labelweights = get_labelweights()
    seg_labelweights = seg_labelweights.to(device)
    compl_labelweights = compl_labelweights.to(device)

    # ===== Training loop =====
    for epoch in range(training_epoch, (config.TRAIN.MAX_EPOCHS)):
        update_level(config, epoch) # Updates config.GENERAL.LEVEL
        pbar = tqdm(train_dataloader)
        train_writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)
        train_writer.add_scalar('epoch', epoch, iteration)
        
        # Training
        model.train()
        for i, batch in enumerate(pbar):
            model.train()
            optimizer.zero_grad()
            # Get tensors from batch
            _, complet_inputs, _, _ = batch

            # forward pass
            _, losses = model(complet_inputs, seg_labelweights, compl_labelweights)
            total_loss: torch.Tensor = 0.0
            total_loss = losses["occupancy_64"] * config.MODEL.OCCUPANCY_64_WEIGHT + losses["semantic_64"]*config.MODEL.SEMANTIC_64_WEIGHT
            if config.MODEL.SEG_HEAD:
                total_loss += losses["pc_seg"]*config.MODEL.PC_SEG_WEIGHT
            if config.GENERAL.LEVEL == "128" or config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                total_loss += losses["occupancy_128"] * config.MODEL.OCCUPANCY_128_WEIGHT + losses["semantic_128"]*config.MODEL.SEMANTIC_128_WEIGHT 
            if config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                total_loss += losses["occupancy_256"]*config.MODEL.OCCUPANCY_256_WEIGHT 
            if config.GENERAL.LEVEL == "FULL":
                total_loss += losses["semantic_256"]*config.MODEL.SEMANTIC_256_WEIGHT

            # backward pass and learning step
            if torch.is_tensor(total_loss):
                total_loss.backward()
                optimizer.step()

                # Logging
                log_msg = {"epoch": epoch}
                log_msg["level"] = config.GENERAL.LEVEL
                for k, v in losses.items():
                    if "256" in k:
                        train_writer.add_scalar('train_256/'+k, v.detach().cpu(), iteration)
                    elif "128" in k:
                        train_writer.add_scalar('train_128/'+k, v.detach().cpu(), iteration)
                    elif "64" in k:
                        train_writer.add_scalar('train_64/'+k, v.detach().cpu(), iteration)
                if config.MODEL.SEG_HEAD:
                    train_writer.add_scalar('train/seg_pc', losses["pc_seg"].detach().cpu(), iteration)    
                log_msg["total_loss"] = total_loss.item()
                pbar.set_postfix(log_msg)
            
            train_writer.add_scalar('train/total_loss', total_loss.detach().cpu(), iteration)
            iteration += 1
            del batch, complet_inputs, total_loss, losses
            torch.cuda.empty_cache()
        # Learning rate scheduler step
        lr_scheduler.step()
        
        # Log memory
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        train_writer.add_scalar('memory/free', info.free, epoch)

        # Save checkpoint
        if epoch % config.TRAIN.CHECKPOINT_PERIOD == 0: # and (config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL"):
            torch.save(model.state_dict(), experiment_dir + "/model{}-{}.pth".format(config.GENERAL.LEVEL, epoch))


        # Evaluation
        if epoch % config.TRAIN.EVAL_PERIOD == 0 and config.GENERAL.LEVEL != "256":
            model.eval()
            log_images = {}
            with torch.no_grad():
                for dataloader_name, dataloader in val_dataloaders.items():
                    seg_evaluators = {"64": iouEval(config.SEGMENTATION.NUM_CLASSES, []),
                                      "128": iouEval(config.SEGMENTATION.NUM_CLASSES, []),  
                                      "256": iouEval(config.SEGMENTATION.NUM_CLASSES, []),
                                      "seg": iouEval(config.SEGMENTATION.NUM_CLASSES, [])}

                    
                    log_images[dataloader_name] = []
                                    
                    print("\n------Evaluating {} set on {} samples------".format(dataloader_name, len(dataloader)))
                    if config.GENERAL.LEVEL == "64":
                        levels = ["64"]
                    elif config.GENERAL.LEVEL == "128":
                        levels = ["64","128"] # ["64", "128"]
                    elif config.GENERAL.LEVEL == "FULL":
                        levels = ["64", "128","256"] #["64", "128", "256"]
                    if config.MODEL.SEG_HEAD:
                        levels.append("seg")
                    for i, batch in enumerate(tqdm(dataloader)):
                        _, complet_inputs, _, _ = batch
                        results = model.inference(complet_inputs)
                    
                        # log images of BEVs to tensorboard
                        if i in eval_imgs_idxs:
                            # input pc
                            input_coords = collect(complet_inputs,"complet_coords").squeeze() # TODO: only works for batch size 1
                            input_remission = collect(complet_inputs,"complet_features")
                            voxels = torch.zeros((1,256,256,32)).int().cuda()
                            voxels[input_coords[:,0].long(),input_coords[:,1].long(),input_coords[:,2].long(),input_coords[:,3].long()]=((input_remission[0]*126.0).int()+1)
                            input_bev = get_bev(voxels)
                            input_bev = input_to_cmap2d(input_bev)
                            log_images[dataloader_name].append((input_bev[0]))

                            bev_labels = collect(complet_inputs, "bev_labels")
                            bev_gt = labels_to_cmap2d(bev_labels)
                            log_images[dataloader_name].append((bev_gt[0]))
                        
                        # iou for pointcloud segmentation
                        if config.MODEL.SEG_HEAD:
                            pc_seg_pred = results['pc_seg']
                            pc_seg_pred = F.softmax(pc_seg_pred, dim=1)
                            pc_seg_pred = torch.argmax(pc_seg_pred, dim=1).cpu().detach().numpy()
                            pc_seg_gt = collect(complet_inputs, "seg_labels").cpu().detach().numpy()
                            pc_seg_pred = pc_seg_pred[pc_seg_gt != -100]
                            pc_seg_gt = pc_seg_gt[pc_seg_gt != -100]
                            seg_evaluators["seg"].addBatch(pc_seg_pred.astype(int)+1, pc_seg_gt.astype(int)+1)

                        for level in levels:
                            if level=="seg": # this doesn't apply for pointcloud segmentation
                                continue
                            semantic_gt = collect(complet_inputs,"complet_labels_{}".format(level)) #.long()
                            occupancy_gt = collect(complet_inputs,"complet_occupancy_{}".format(level)) #.long()
                            semantic_labels = results['semantic_labels_{}'.format(level)]

                            if level == "64":
                                occupancy_prediction = results['occupancy_{}'.format(level)].squeeze() > 0.5
                                occupancy_gt = occupancy_gt.to('cpu').data.numpy().flatten()
                                occupancy_prediction = occupancy_prediction.to('cpu').data.numpy().flatten()
                                # Add img for logging
                                if i in eval_imgs_idxs and level == config.GENERAL.LEVEL:
                                    semantic_labels_rgb = get_bev(semantic_labels)
                                    semantic_labels_rgb = F.interpolate(semantic_labels_rgb.unsqueeze(0).float(), size=(256,256), mode="nearest")
                                    semantic_labels_rgb = labels_to_cmap2d(semantic_labels_rgb[0].long())
                                    log_images[dataloader_name].append((semantic_labels_rgb[0]))
                                semantic_gt = semantic_gt.to('cpu').data.numpy().flatten()
                                semantic_labels = semantic_labels.to('cpu').data.numpy().flatten()
                                seg_labels_gt = semantic_gt[semantic_gt!=255]
                                semantic_labels = semantic_labels[semantic_gt!=255]
                                seg_evaluators[level].addBatch(semantic_labels.astype(int), seg_labels_gt.astype(int))
                            else:
                                min_coordinate = torch.IntTensor([0, 0, 0])
                                shape = shapes[level]
                                # prediction = results['semantic_prediction_{}'.format(level)]
                                semantic_labels, _, _ = semantic_labels.dense(shape, min_coordinate=min_coordinate)
                                semantic_labels[:,semantic_gt == 255] = 0
                                # Add img for logging
                                if i in eval_imgs_idxs and (level == config.GENERAL.LEVEL or level == "256"):
                                    semantic_labels_rgb = get_bev(semantic_labels[0])
                                    semantic_labels_rgb = F.interpolate(semantic_labels_rgb.unsqueeze(0).float(), size=(256,256), mode="nearest")
                                    semantic_labels_rgb = labels_to_cmap2d(semantic_labels_rgb[0].long())
                                    log_images[dataloader_name].append((semantic_labels_rgb[0]))
                                    
                                semantic_labels = np.uint16(semantic_labels.to("cpu").detach().cpu().numpy()).flatten()
                                semantic_gt = np.uint16(semantic_gt.detach().cpu().numpy()).flatten()
                                semantic_labels = semantic_labels[semantic_gt!=255]
                                
                                
                                occupancy_prediction = results['occupancy_{}'.format(level)]
                                occupancy_prediction, _, _ = occupancy_prediction.dense(shape, min_coordinate=min_coordinate)
                                occupancy_prediction = np.uint16(occupancy_prediction.to("cpu").detach().cpu().numpy()).flatten()
                                occupancy_gt = np.uint16(occupancy_gt.detach().cpu().numpy()).flatten()
                                occupancy_gt = occupancy_gt[semantic_gt!=255]
                                occupancy_prediction = occupancy_prediction[semantic_gt!=255]
                                semantic_gt = semantic_gt[semantic_gt!=255]
                                seg_evaluators[level].addBatch(semantic_labels.astype(int), semantic_gt.astype(int))
                        del batch, complet_inputs, results, semantic_gt, occupancy_gt, semantic_labels, occupancy_prediction
                        torch.cuda.empty_cache()
                
                    # log eval results

                    for level in levels:
                        print("Evaluating level: {}".format(level))
                        _, class_jaccard = seg_evaluators[level].getIoU()
                        m_jaccard = class_jaccard[1:].mean()
                        ignore = [0]
                        for i, jacc in enumerate(class_jaccard):
                            if i not in ignore:
                                writers[dataloader_name].add_scalar('eval-{}/{}'.format(level,seg_label_to_cat[i]), jacc*100, epoch)
                                # print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                                #     i=i, class_str=seg_label_to_cat[i], jacc=jacc*100))
                        print('{} point avg class IoU: {}'.format(dataloader_name,m_jaccard*100))
                        
                        # compute remaining metrics.
                        conf = seg_evaluators[level].get_confusion()
                        precision = np.sum(conf[1:,1:]) / (np.sum(conf[1:,:]) + epsilon)
                        recall = np.sum(conf[1:,1:]) / (np.sum(conf[:,1:]) + epsilon)
                        acc_cmpltn = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0,0])

                        print("Precision =\t" + str(np.round(precision * 100, 2)) + '\n' +
                                "Recall =\t" + str(np.round(recall * 100, 2)) + '\n' +
                                "IoU Cmpltn =\t" + str(np.round(acc_cmpltn * 100, 2)) + '\n')   
                        
                        writers[dataloader_name].add_scalar('eval-{}/mIoU'.format(level), m_jaccard*100, epoch)
                        writers[dataloader_name].add_scalar('eval-{}/precision'.format(level), precision*100, epoch)
                        writers[dataloader_name].add_scalar('eval-{}/recall'.format(level), recall*100, epoch)
                        writers[dataloader_name].add_scalar('eval-{}/acc_cmpltn'.format(level), acc_cmpltn*100, epoch)

                    # log bev images:
                    imgs = torch.Tensor(log_images[dataloader_name])
                    num_rows = 4 if config.MODEL.UNET2D else 3
                    grid_imgs = torchvision.utils.make_grid(imgs, nrow=num_rows)
                    writers[dataloader_name].add_image('eval/bev', grid_imgs, epoch)

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

            

    
