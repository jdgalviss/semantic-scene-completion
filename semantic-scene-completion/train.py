import torch
from configs import config, LABEL_TO_NAMES
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import nvidia_smi
from tqdm import tqdm
from torch.nn import functional as F
import torchvision
from model import DSKDLoss

from model import MyModel
from structures import collect
from semantic_kitti_dataset import get_labelweights
from utils import re_seed, labels_to_cmap2d, get_bev, input_to_cmap2d, get_dataloaders, update_level, CosineAnnealingWarmupRestarts
from utils import create_new_experiment_folder, save_config
from evaluation import iouEval

epsilon = np.finfo(np.float32).eps
device = torch.device("cuda:0")
eval_imgs_idxs = [100,200,300,400,500,600,700,800,10,250,370,420,580,600] # Randomly chosen samples from which bev images will be logged in tensorboard
# eval_imgs_idxs = [0,4,6,8,10,12,14,16,18]

def main():
    re_seed(config.GENERAL.MANUAL_SEED)
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
    
    optimizer = torch.optim.Adam(model.parameters(), config.SOLVER.BASE_LR,
                                        betas=(config.SOLVER.BETA_1, config.SOLVER.BETA_2),
                                        weight_decay=config.SOLVER.WEIGHT_DECAY)
    
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.SOLVER.LR_DECAY_RATE)
    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=len(train_dataloader)*20, cycle_mult=0.5, max_lr=config.SOLVER.BASE_LR, min_lr=config.SOLVER.BASE_LR/5.0, warmup_steps=int(len(train_dataloader)/5), gamma=0.5)

    if config.MODEL.DISTILLATION:
        model_teacher = MyModel(is_teacher=True).to(device)
        distillation_criteria = DSKDLoss()
        
    # Load checkpoint
    if config.GENERAL.CHECKPOINT_PATH is not None:
        model.load_state_dict(torch.load(config.GENERAL.CHECKPOINT_PATH))
        training_epoch = int(config.GENERAL.CHECKPOINT_PATH.split('-')[-1].split('.')[0]) + 1
        print("TRAINING_EPOCH: ", training_epoch)
    else:
        training_epoch = 0
        if config.MODEL.SEG_HEAD and config.SEGMENTATION.CHECKPOINT is not None and config.SEGMENTATION.SEG_MODEL == "vanilla":
            model_seg_checkpoint = MyModel().cuda()
            model_seg_checkpoint.load_state_dict(torch.load(config.SEGMENTATION.CHECKPOINT))
            model.seg_model.load_state_dict(model_seg_checkpoint.seg_model.state_dict())
            print("Loaded pretrained segmentation model from: ", config.SEGMENTATION.CHECKPOINT)
            del model_seg_checkpoint

    if config.MODEL.DISTILLATION:
        try:
            model_teacher.load_state_dict(torch.load(config.GENERAL.TEACHER_CHECKPOINT_PATH))
            model_teacher.eval()
            print("LOADED DISTILLATION TEACHER MODEL")
        except:
            print("Teacher model checkpoint not found, can't apply distillation")
            return 0
        
    
    
    iteration = training_epoch * len(train_dataloader)
    seg_label_to_cat = LABEL_TO_NAMES #TODO: replace

    # Number of Parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: ", pytorch_total_params)
    print("Total trainable params: ", pytorch_total_trainable_params)

    # Setup logging
    experiment_dir = create_new_experiment_folder(config.GENERAL.OUT_DIR, config.GENERAL.EXPERIMENT_NAME)
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
    consecutive_fails = 0
    # ===== Training loop =====
    for epoch in range(training_epoch, (config.TRAIN.MAX_EPOCHS)):
        update_level(config, epoch) # Updates config.GENERAL.LEVEL
        pbar = tqdm(train_dataloader)
        # train_writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)
        train_writer.add_scalar('epoch', epoch, iteration)
        
        # Training
        for i, batch in enumerate(pbar):
            # if cosine_annealing:
            train_writer.add_scalar('train/lr', lr_scheduler.get_lr()[0], iteration)
            # Learning rate scheduler step cosine annealing
            lr_scheduler.step()
            model.train()
            _, complet_inputs, _, _ = batch
            optimizer.zero_grad()

            if config.MODEL.DISTILLATION:
                # forward pass teacher multi-pc model
                try:
                    with torch.no_grad():
                        _, _, features_teacher, _ = model_teacher(complet_inputs, seg_labelweights, compl_labelweights)
                except Exception as e:
                    print(e, "Error in forward pass teacher: ", iteration)
                    consecutive_fails += 1
                    if consecutive_fails > 100:
                        print("Too many consecutive fails, exiting")
                        return
                    del complet_inputs
                    torch.cuda.empty_cache()
                    continue
            total_loss: torch.Tensor = 0.0

            # forward pass single-pc model
            # try:
            _, losses, features, sigma = model(complet_inputs, seg_labelweights, compl_labelweights)
            # except Exception as e:
            #     print(e, "Error in forward pass: ", iteration)
            #     consecutive_fails += 1
            #     if consecutive_fails > 100:
            #         print("Too many consecutive fails, exiting")
            #         return
            #     del complet_inputs
            #     torch.cuda.empty_cache()
            #     continue
            consecutive_fails = 0

            # Compute weighted loss
            if config.TRAIN.UNCERTAINTY_LOSS:
                factor_compl = 1.0 / (sigma[1]**2)
                total_loss = factor_compl[0] * losses["occupancy_64"] + 2 * torch.log(sigma[1][0]) + \
                            factor_compl[1] * losses["semantic_64"] + 2 * torch.log(sigma[1][1]) 
                if config.MODEL.SEG_HEAD:
                    factor_seg = 1.0 / (sigma[0]**2)
                    total_loss += factor_seg[0] * 0.1 * losses["pc_seg"] + 2 * torch.log(sigma[0][0])
                    train_writer.add_scalar('factors/seg_pc', factor_seg.detach().cpu(), iteration)
                if config.GENERAL.LEVEL == "128" or config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                    total_loss += factor_compl[2] * losses["occupancy_128"] + 2 * torch.log(sigma[1][2]) + \
                                    factor_compl[3] * losses["semantic_128"] + 2 * torch.log(sigma[1][3])  
                if config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                    total_loss += factor_compl[4] * losses["occupancy_256"] + 2 * torch.log(sigma[1][4])
                if config.GENERAL.LEVEL == "FULL":
                    total_loss += factor_compl[5] * losses["semantic_256"] + 2 * torch.log(sigma[1][5])
                train_writer.add_scalar('factors/complt_64', factor_compl[0].detach().cpu(), iteration)
                train_writer.add_scalar('factors/seg_64', factor_compl[1].detach().cpu(), iteration)
                train_writer.add_scalar('factors/complt_128', factor_compl[2].detach().cpu(), iteration)
                train_writer.add_scalar('factors/seg_128', factor_compl[3].detach().cpu(), iteration)
                train_writer.add_scalar('factors/complt_256', factor_compl[4].detach().cpu(), iteration)
                train_writer.add_scalar('factors/seg_256', factor_compl[5].detach().cpu(), iteration)
            else:
                total_loss = losses["occupancy_64"] * config.MODEL.OCCUPANCY_64_WEIGHT + losses["semantic_64"]*config.MODEL.SEMANTIC_64_WEIGHT
                if config.MODEL.SEG_HEAD:
                    total_loss += losses["pc_seg"]*config.MODEL.PC_SEG_WEIGHT
                if config.GENERAL.LEVEL == "128" or config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                    total_loss += losses["occupancy_128"] * config.MODEL.OCCUPANCY_128_WEIGHT + losses["semantic_128"]*config.MODEL.SEMANTIC_128_WEIGHT 
                if config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
                    total_loss += losses["occupancy_256"]*config.MODEL.OCCUPANCY_256_WEIGHT 
                if config.GENERAL.LEVEL == "FULL":
                    total_loss += losses["semantic_256"]*config.MODEL.SEMANTIC_256_WEIGHT

            if config.MODEL.DISTILLATION:
                distillation_loss = distillation_criteria(features, features_teacher)
                total_loss += distillation_loss * config.MODEL.DISTILLATION_WEIGHT

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
                if config.MODEL.DISTILLATION:
                    train_writer.add_scalar('train/distillation', distillation_loss.detach().cpu(), iteration)
                log_msg["total_loss"] = total_loss.item()
                pbar.set_postfix(log_msg)
            train_writer.add_scalar('train/total_loss', total_loss.detach().cpu(), iteration)
            iteration += 1
            del batch, complet_inputs, total_loss, losses, features
            if config.MODEL.DISTILLATION:
                del features_teacher
            torch.cuda.empty_cache()
        # Learning rate scheduler step
        # lr_scheduler.step()
        
        # Log memory
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        train_writer.add_scalar('memory/free', info.free, epoch)

        # Save checkpoint
        if epoch % config.TRAIN.CHECKPOINT_PERIOD == 0 and (config.GENERAL.LEVEL == "FULL"):
            torch.save(model.state_dict(), experiment_dir + "/model{}-{}.pth".format(config.GENERAL.LEVEL, epoch))

        # Evaluation
        if epoch % config.TRAIN.EVAL_PERIOD == 0 and config.GENERAL.LEVEL != "256":
            model.eval()
            log_images = {}
            with torch.no_grad():
                # Evaluation is done on training subset and eval set to detect overfitting
                for dataloader_name, dataloader in val_dataloaders.items():
                    # Evaluators on every level
                    seg_evaluators = {"64": iouEval(config.SEGMENTATION.NUM_CLASSES, []),
                                      "128": iouEval(config.SEGMENTATION.NUM_CLASSES, []),  
                                      "256": iouEval(config.SEGMENTATION.NUM_CLASSES, []),
                                      "seg": iouEval(config.SEGMENTATION.NUM_CLASSES, [])}
                    log_images[dataloader_name] = []
                    print("\n------Evaluating {} set on {} samples------".format(dataloader_name, len(dataloader)))
                    # For each hierarchy level, define which other scales are available for multiscale evaluation
                    if config.GENERAL.LEVEL == "64":
                        levels = ["64"]
                    elif config.GENERAL.LEVEL == "128":
                        levels = ["64","128"]
                    elif config.GENERAL.LEVEL == "FULL":
                        levels = ["64", "128","256"]
                    if config.MODEL.SEG_HEAD:
                        levels.append("seg")
                    
                    # Evaluation loop
                    for i, batch in enumerate(tqdm(dataloader)):
                        _, complet_inputs, _, _ = batch
                        try:
                            results = model.inference(complet_inputs)
                        except Exception as e:
                            print(e, "Error in forward pass - evaluation: ", iteration)
                            continue
                    
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
                            # gt bev
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
                    num_rows = 3
                    grid_imgs = torchvision.utils.make_grid(imgs, nrow=num_rows)
                    writers[dataloader_name].add_image('eval/bev', grid_imgs, epoch)
                    del imgs, grid_imgs
                del log_images
                torch.cuda.empty_cache()

            
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
