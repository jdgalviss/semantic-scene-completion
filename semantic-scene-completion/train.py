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
from utils import re_seed, labels_to_cmap2d, get_bev, input_to_cmap2d
from structures import collect
from evaluation import iouEval
from model import get_sparse_values
import nvidia_smi
from model import UNet
import torchvision

epsilon = np.finfo(np.float32).eps
device = torch.device("cuda:0")
eval_imgs_idxs = [10,111,215,420,541]

def main():
    re_seed(0)
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # shapes = {"256": torch.Size([config.TRAIN.BATCH_SIZE, 1, 256, 256, 32]), "128": torch.Size([config.TRAIN.BATCH_SIZE, 1, 128, 128, 16]), "64": torch.Size([config.TRAIN.BATCH_SIZE, 1, 64, 64, 8])}
    shapes = {"256": torch.Size([1, 1, 256, 256, 32]), "128": torch.Size([1, 1, 128, 128, 16]), "64": torch.Size([1, 1, 64, 64, 8])}

    # Datasets and dataloaders
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
        batch_size=1,
        collate_fn=Merge,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        # worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )
    trainval_dataloader = torch.utils.data.DataLoader(
        trainval_dataset,
        batch_size=1,
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

    # Hybrid 3D Completion UNet
    model = MyModel(num_output_channels=config.MODEL.NUM_OUTPUT_CHANNELS, unet_features=config.MODEL.UNET_FEATURES).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), config.SOLVER.BASE_LR,
                                          betas=(config.SOLVER.BETA_1, config.SOLVER.BETA_2),
                                          weight_decay=config.SOLVER.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.SOLVER.LR_DECAY_RATE)
    if config.GENERAL.CHECKPOINT_PATH is not None:
        model.load_state_dict(torch.load(config.GENERAL.CHECKPOINT_PATH))
        training_epoch = int(config.GENERAL.CHECKPOINT_PATH.split('-')[-1].split('.')[0]) + 1
        print("TRAINING_EPOCH: ", training_epoch)
    else:
        training_epoch = 0
    model = model.to(device)
    
    model.train()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: ", pytorch_total_params)
    print("Total trainable params: ", pytorch_total_trainable_params)

    steps_schedule = config.TRAIN.STEPS
    iteration = training_epoch * len(train_dataloader)
    seg_label_to_cat = train_dataset.label_to_names
    
    # 2D BEV Semantic Completion Network
    if config.MODEL.UNET2D:
        unet2d_model = UNet(n_channels=7, n_classes=config.SEGMENTATION.NUM_CLASSES, num_output_features=config.MODEL.NUM_OUTPUT_CHANNELS).to(device)
        optimizer2d = torch.optim.Adam(unet2d_model.parameters(), config.SOLVER.BASE_LR,
                                          betas=(config.SOLVER.BETA_1, config.SOLVER.BETA_2),
                                          weight_decay=config.SOLVER.WEIGHT_DECAY)
        lr_scheduler2d = torch.optim.lr_scheduler.ExponentialLR(optimizer2d, gamma=config.SOLVER.LR_DECAY_RATE)
        
        unet2d_model.train()
        pytorch_total_params = sum(p.numel() for p in unet2d_model.parameters())
        pytorch_total_trainable_params = sum(p.numel() for p in unet2d_model.parameters() if p.requires_grad)
        print("Total params 2D model: ", pytorch_total_params)
        print("Total trainable params 2D model: ", pytorch_total_trainable_params)

    # # Optimizer
    # optimizer = torch.optim.Adam(list(model.parameters())+list(unet2d_model.parameters()), config.SOLVER.BASE_LR,
    #                                       betas=(config.SOLVER.BETA_1, config.SOLVER.BETA_2),
    #                                       weight_decay=config.SOLVER.WEIGHT_DECAY)


    # Setup logging
    experiment_dir = create_new_experiment_folder(config.GENERAL.OUT_DIR)
    save_config(config, experiment_dir)
    train_writer = SummaryWriter(log_dir=str(experiment_dir + "/train"))

    if config.GENERAL.OVERFIT:
        writers = {"trainval": train_writer}
    else:   
        eval_writer = SummaryWriter(log_dir=str(experiment_dir + "/eval"))
        writers = {"val": eval_writer, "trainval": train_writer}


    # ============== Training ==============
    for epoch in range(training_epoch, (config.TRAIN.MAX_EPOCHS)):
        # ============== Evaluation ==============
        if epoch>=steps_schedule[0]:
            if epoch>=steps_schedule[1]:
                if epoch>=steps_schedule[2]:
                    config.GENERAL.LEVEL = "FULL"
                    config.TRAIN.EVAL_PERIOD = 2 if epoch >= (steps_schedule[2]+1) else 1
                else:
                    config.GENERAL.LEVEL = "256"
            else:
                config.GENERAL.LEVEL = "128"
                
        if epoch % config.TRAIN.EVAL_PERIOD == 0 and config.GENERAL.LEVEL != "256":
            model.eval()
            if config.MODEL.UNET2D:
                unet2d_model.eval()
            log_images = {}
            with torch.no_grad():
                for dataloader_name, dataloader in val_dataloaders.items():
                    seg_evaluators = {"64": iouEval(config.SEGMENTATION.NUM_CLASSES, []),
                                      "128": iouEval(config.SEGMENTATION.NUM_CLASSES, []),  
                                      "256": iouEval(config.SEGMENTATION.NUM_CLASSES, []),}

                    occ_evaluators = {"64": iouEval(2, []),
                                        "128": iouEval(2, []),  
                                        "256": iouEval(2, []),}
                    log_images[dataloader_name] = []
                                    
                    print("\n------Evaluating {} set on {} samples------".format(dataloader_name, len(dataloader)))
                    if config.GENERAL.LEVEL == "64":
                        levels = ["64"]
                    elif config.GENERAL.LEVEL == "128":
                        levels = ["128"] # ["64", "128"]
                    elif config.GENERAL.LEVEL == "FULL":
                        levels = ["256"] #["64", "128", "256"]
                    for i, batch in enumerate(tqdm(dataloader)):
                        _, complet_inputs, _, _ = batch
                        # try:
                        if config.MODEL.UNET2D:
                            bev_pred, f1, f2, f3, f4, f5 = unet2d_model.inference(collect(complet_inputs, "input2d"))
                            features2D = [f2,f3,f4,f5]
                        else:
                            features2D = None
                        results = model.inference(complet_inputs, features2D=features2D)
                        # except Exception as e:
                        #     print(e, "Error in inference: ", iteration)
                        #     del complet_inputs
                        #     torch.cuda.empty_cache()
                        #     continue
                        # Semantic Eval
                        if i in eval_imgs_idxs and config.MODEL.UNET2D:
                            #TODO: convert output to image
                            # input
                            input_coords = collect(complet_inputs,"complet_coords").squeeze() # TODO: only works for batch size 1
                            input_remission = collect(complet_inputs,"complet_features")
                            voxels = torch.zeros((1,256,256,32)).int().cuda()
                            voxels[input_coords[:,0].long(),input_coords[:,1].long(),input_coords[:,2].long(),input_coords[:,3].long()]=((input_remission[0]*126.0).int()+1)
                            input_bev = get_bev(voxels)
                            input_bev = input_to_cmap2d(input_bev)
                            log_images[dataloader_name].append((input_bev[0]))

                            # gt and prediction
                            bev_labels = collect(complet_inputs, "bev_labels")
                            bev_pred[bev_labels==255] = 0
                            bev_pred_rgb = labels_to_cmap2d(bev_pred)
                            log_images[dataloader_name].append((bev_pred_rgb[0]))
                            bev_gt = labels_to_cmap2d(bev_labels)
                            log_images[dataloader_name].append((bev_gt[0]))

                        

                        for level in levels:
                            semantic_gt = collect(complet_inputs,"complet_labels_{}".format(level)) #.long()
                            occupancy_gt = collect(complet_inputs,"complet_occupancy_{}".format(level)) #.long()
                            semantic_labels = results['semantic_labels_{}'.format(level)]

                            if level == "64":
                                occupancy_prediction = results['occupancy_{}'.format(level)].squeeze() > 0.5
                                occupancy_gt = occupancy_gt.to('cpu').data.numpy().flatten()
                                occupancy_prediction = occupancy_prediction.to('cpu').data.numpy().flatten()
                                occ_evaluators[level].addBatch(occupancy_prediction.astype(int), occupancy_gt.astype(int))
                                # Add img for logging
                                if i in eval_imgs_idxs:
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
                                if i in eval_imgs_idxs:
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
                                occ_evaluators[level].addBatch(occupancy_prediction.astype(int), occupancy_gt.astype(int))
                                seg_evaluators[level].addBatch(semantic_labels.astype(int), semantic_gt.astype(int))
                        del batch, complet_inputs, results, semantic_gt, occupancy_gt, semantic_labels, occupancy_prediction
                        torch.cuda.empty_cache()

                    for level in levels:
                        print("\nEvaluating level: {}".format(level))
                        occ_miou, occ_iou = occ_evaluators[level].getIoU()
                        _, class_jaccard = seg_evaluators[level].getIoU()
                        m_jaccard = class_jaccard[1:].mean()
                        ignore = [0]
                        for i, jacc in enumerate(class_jaccard):
                            if i not in ignore:
                                writers[dataloader_name].add_scalar('eval-{}/{}'.format(level,seg_label_to_cat[i]), jacc*100, epoch)
                                print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                                    i=i, class_str=seg_label_to_cat[i], jacc=jacc*100))
                        print('\n{} point avg class IoU: {}'.format(dataloader_name,m_jaccard*100))
                        print('{} point avg occupancy IoU: {}'.format(dataloader_name,occ_miou*100))
                        
                        # compute remaining metrics.
                        conf = seg_evaluators[level].get_confusion()
                        precision = np.sum(conf[1:,1:]) / (np.sum(conf[1:,:]) + epsilon)
                        recall = np.sum(conf[1:,1:]) / (np.sum(conf[:,1:]) + epsilon)
                        acc_cmpltn = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0,0])

                        print("Precision =\t" + str(np.round(precision * 100, 2)) + '\n' +
                                "Recall =\t" + str(np.round(recall * 100, 2)) + '\n' +
                                "IoU Cmpltn =\t" + str(np.round(acc_cmpltn * 100, 2)) + '\n')   
                        
                        writers[dataloader_name].add_scalar('eval-{}/mIoU'.format(level), m_jaccard*100, epoch)
                        writers[dataloader_name].add_scalar('eval-{}/occ_mIoU'.format(level), occ_miou*100, epoch)
                        writers[dataloader_name].add_scalar('eval-{}/precision'.format(level), precision*100, epoch)
                        writers[dataloader_name].add_scalar('eval-{}/recall'.format(level), recall*100, epoch)
                        writers[dataloader_name].add_scalar('eval-{}/acc_cmpltn'.format(level), acc_cmpltn*100, epoch)

                    # log bev images:
                    imgs = torch.Tensor(log_images[dataloader_name])
                    num_rows = 4 if config.MODEL.UNET2D else 1
                    grid_imgs = torchvision.utils.make_grid(imgs, nrow=num_rows)
                    writers[dataloader_name].add_image('eval/bev', grid_imgs, epoch)


                    
                    torch.cuda.empty_cache()

        # # ============== Training ==============
        
        #log learning rate
        train_writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)
    

        # with tqdm(total=len(train_dataloader)) as pbar:
        # for i, batch in enumerate(train_dataloader):
        pbar = tqdm(train_dataloader)
        consecutive_fails = 0
        # val_dataloader_iterator = iter(val_dataloader) # VAL: Comment for no val
        for i, batch in enumerate(pbar):
            model.train()
            optimizer.zero_grad()
            if config.MODEL.UNET2D:
                unet2d_model.train()
                optimizer2d.zero_grad()
            
            # print("batch len: ", len(batch))

            # Get tensors from batch
            _, complet_inputs, _, _ = batch
            compl_labelweights = torch.Tensor(train_dataset.compl_labelweights).cuda()

            if config.MODEL.UNET2D:
                _, loss2d, _, f2, f3, f4, f5 = unet2d_model(collect(complet_inputs, "input2d"), collect(complet_inputs, "bev_labels"),compl_labelweights)
                loss2d.backward()
                optimizer2d.step()
                features2D = [f2.detach(),f3.detach(),f4.detach(),f5.detach()]
                # features2D = [f2,f3,f4,f5]
                train_writer.add_scalar('bev/loss', loss2d.detach().cpu(), iteration)

            else:
                features2D = None
            # seg_labelweights = torch.Tensor(train_dataset.seg_labelweights).cuda()
            
            # Forward pass through model
            # try:
            losses, _ = model(complet_inputs, compl_labelweights, features2D=features2D)
            # except Exception as e:
            #     print(e, "Error in forward pass: ", iteration)
            #     consecutive_fails += 1
            #     if consecutive_fails > 20:
            #         print("Too many consecutive fails, exiting")
            #         return
            #     del complet_inputs
            #     torch.cuda.empty_cache()
            #     continue
            consecutive_fails = 0
            total_loss: torch.Tensor = 0.0
            total_loss = losses["occupancy_64"] * config.MODEL.OCCUPANCY_64_WEIGHT + losses["semantic_64"]*config.MODEL.SEMANTIC_64_WEIGHT # + losses["gen_64"]*config.MODEL.GEN_64_WEIGHT + losses["semantic2D_64"]*config.MODEL.SEMANTIC2D_64_WEIGHT
            # if config.MODEL.UNET2D:
            #     total_loss += loss2d*0.01
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
            del batch, complet_inputs, total_loss, losses
            
            
            

            

            # Minkowski Engine recommendation
            # torch.cuda.empty_cache()

            # # ============== Validation ==============
            # model.eval()
            # with torch.no_grad():
            #     try:
            #         val_batch = next(val_dataloader_iterator)
            #     except StopIteration:
            #         val_dataloader_iterator = iter(val_dataloader)
            #         val_batch = next(val_dataloader_iterator)
            #     # Get tensors from batch
            #     _, complet_inputs, _, _ = val_batch
            #     # seg_labelweights = torch.Tensor(train_dataset.seg_labelweights).cuda()
            #     compl_labelweights = torch.Tensor(train_dataset.compl_labelweights).cuda()
                
            #     # Forward pass through model
            #     try:
            #         losses, _ = model(complet_inputs, compl_labelweights, is_train_mod=False)
            #     except Exception as e:
            #         print(e, "Error in val forward pass: ", iteration)
            #         del complet_inputs, val_batch
            #         torch.cuda.empty_cache()
            #         continue
            #     total_loss: torch.Tensor = 0.0
            #     total_loss = losses["occupancy_64"] * config.MODEL.OCCUPANCY_64_WEIGHT + losses["semantic_64"]*config.MODEL.SEMANTIC_64_WEIGHT 
            #     if config.GENERAL.LEVEL == "128" or config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
            #         total_loss += losses["occupancy_128"] * config.MODEL.OCCUPANCY_128_WEIGHT + losses["semantic_128"]*config.MODEL.SEMANTIC_128_WEIGHT 
            #     if config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL":
            #         total_loss += losses["occupancy_256"]*config.MODEL.OCCUPANCY_256_WEIGHT 
            #     if config.GENERAL.LEVEL == "FULL":
            #         total_loss += losses["semantic_256"]*config.MODEL.SEMANTIC_256_WEIGHT
            #     if torch.is_tensor(total_loss):
            #         for k, v in losses.items():
            #             # log_msg[k] = v.item()
            #             # log_msg += "{}: {:.4f}, ".format(k, v)
            #             if "256" in k:
            #                 eval_writer.add_scalar('train_256/'+k, v.detach().cpu(), iteration)
            #             elif "128" in k:
            #                 eval_writer.add_scalar('train_128/'+k, v.detach().cpu(), iteration)
            #             elif "64" in k:
            #                 eval_writer.add_scalar('train_64/'+k, v.detach().cpu(), iteration)

            #     eval_writer.add_scalar('train/total_loss', total_loss.detach().cpu(), iteration)
            #     del val_batch, complet_inputs, total_loss
            
            # # Log memory
            # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            # eval_writer.add_scalar('memory/free', info.free, float(epoch) + float(iteration)/float(len(val_dataloader)))

            iteration += 1
            torch.cuda.empty_cache()
            # ========================================
        train_writer.add_scalar('epoch', epoch, iteration)
        # Log memory
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        train_writer.add_scalar('memory/free', info.free, epoch)

        # Save checkpoint
        if epoch % config.TRAIN.CHECKPOINT_PERIOD == 0 and (config.GENERAL.LEVEL == "256" or config.GENERAL.LEVEL == "FULL"):
            torch.save(model.state_dict(), experiment_dir + "/model{}-{}.pth".format(config.GENERAL.LEVEL, epoch+1))
            if config.MODEL.UNET2D:
                torch.save(unet2d_model.state_dict(), experiment_dir + "/model2d{}-{}.pth".format(config.GENERAL.LEVEL, epoch+1))
            torch.save(model.state_dict(), experiment_dir + "/model{}-{}.pth".format(config.GENERAL.LEVEL, epoch+1))


        # Learning rate scheduler step
        lr_scheduler.step()
        if config.MODEL.UNET2D:
            lr_scheduler2d.step()
            
    torch.save(model.state_dict(), experiment_dir + "/model.pth")



if __name__ == '__main__':
    # import debugpy
    # print("waiting for debugger to attach...")
    # debugpy.listen(('131.159.98.103', 5678))
    # debugpy.wait_for_client()
    # print("debugger attached")
    # Arguments
    parser = argparse.ArgumentParser(description="Semantic Scene Completion")
    parser.add_argument("--config-file", type=str, default="configs/ssc.yaml", required=False)
    parser.add_argument("--output-path", type=str, default="experiments", required=False)

    args = parser.parse_args()
    config.GENERAL.OUT_DIR = args.output_path
    config.merge_from_file(args.config_file)
    print("\n Training with configuration parameters: \n",config)

    main()

