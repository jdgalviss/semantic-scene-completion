import torch
from configs import config, LABEL_TO_NAMES
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F

from model import MyModel
from structures import collect
from semantic_kitti_dataset import get_labelweights
from utils import re_seed, get_dataloaders
from evaluation import iouEval
torch.autograd.set_detect_anomaly(True)

epsilon = np.finfo(np.float32).eps
device = torch.device("cuda:0")
eval_imgs_idxs = [100,200,300,400,500,600,700,800,10,250,370,420,580,600]
# eval_imgs_idxs = [0,4,6,8,10,12,14,16,18]
    

def main():
    re_seed(0)
    config.GENERAL.LEVEL = "FULL"

    # completion different voxel tensors sizes
    shapes = {"256": torch.Size([1, 1, 256, 256, 32]), "128": torch.Size([1, 1, 128, 128, 16]), "64": torch.Size([1, 1, 64, 64, 8])}

    # Get dataloaders
    _, val_dataloader, _ = get_dataloaders(config) 
    val_dataloaders = {"val": val_dataloader}
    
    # SSC Model
    model = MyModel(False).to(device)
    

    # Load checkpoint
    if config.GENERAL.CHECKPOINT_PATH is not None:
        model.load_state_dict(torch.load(config.GENERAL.CHECKPOINT_PATH))

        training_epoch = int(config.GENERAL.CHECKPOINT_PATH.split('-')[-1].split('.')[0]) + 1
        print("TRAINING_EPOCH: ", training_epoch)
    seg_label_to_cat = LABEL_TO_NAMES #TODO: replace

    # Number of Parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: ", pytorch_total_params)
    print("Total trainable params: ", pytorch_total_trainable_params)

    # Setup logging
    # get labelweights
    seg_labelweights, compl_labelweights = get_labelweights()
    seg_labelweights = seg_labelweights.to(device)
    compl_labelweights = compl_labelweights.to(device)
    consecutive_fails = 0
    # ===== Training loop =====
    # for epoch in range(training_epoch, (config.TRAIN.MAX_EPOCHS)):
    # Evaluation
    model.eval()
    log_images = {}
    with torch.no_grad():
        # update_level(config, 100)
        for dataloader_name, dataloader in val_dataloaders.items():
            seg_evaluators = {"64": iouEval(config.SEGMENTATION.NUM_CLASSES, []),
                                "128": iouEval(config.SEGMENTATION.NUM_CLASSES, []),  
                                "256": iouEval(config.SEGMENTATION.NUM_CLASSES, []),
                                "seg": iouEval(config.SEGMENTATION.NUM_CLASSES, [])}
            
                            
            print("\n------Evaluating {} set on {} samples------".format(dataloader_name, len(dataloader)))
            
            levels = ["256"] #["64", "128", "256"]
            if config.MODEL.SEG_HEAD:
                levels.append("seg")
            for i, batch in enumerate(tqdm(dataloader)):
                _, complet_inputs, _, _ = batch
                try:
                    results = model.inference(complet_inputs)
                except Exception as e:
                    print(e, "Error in forward pass - evaluation: ")
                    continue
            
                # log images of BEVs to tensorboard
                
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
                        print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                            i=i, class_str=seg_label_to_cat[i], jacc=jacc*100))
                print('{} point avg class IoU: {}'.format(dataloader_name,m_jaccard*100))
                
                # compute remaining metrics.
                conf = seg_evaluators[level].get_confusion()
                precision = np.sum(conf[1:,1:]) / (np.sum(conf[1:,:]) + epsilon)
                recall = np.sum(conf[1:,1:]) / (np.sum(conf[:,1:]) + epsilon)
                acc_cmpltn = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0,0])

                print("Precision =\t" + str(np.round(precision * 100, 2)) + '\n' +
                        "Recall =\t" + str(np.round(recall * 100, 2)) + '\n' +
                        "IoU Cmpltn =\t" + str(np.round(acc_cmpltn * 100, 2)) + '\n')   

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="Semantic Scene Completion")
    parser.add_argument("--config-file", type=str, default="configs/ssc.yaml", required=False)
    parser.add_argument("--checkpoint", type=str, default="/usr/src/app/semantic-scene-completion/data/modelFULL-19.pth", required=False)

    args = parser.parse_args()
    config.merge_from_file(args.config_file)
    config.GENERAL.CHECKPOINT_PATH = args.checkpoint

    print("\n Evaluating with configuration parameters: \n",config)

    main()

            

    
