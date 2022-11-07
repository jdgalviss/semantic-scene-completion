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

device = torch.device("cuda:0")


def main():
    re_seed(0)
    test_dataset = SemanticKITTIDataset(config, "test",do_overfit=False)
    

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        collate_fn=Merge,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )

    model = MyModel(num_output_channels=config.MODEL.NUM_OUTPUT_CHANNELS, unet_features=config.MODEL.UNET_FEATURES).to(device)
    try:
        model.load_state_dict(torch.load(config.GENERAL.CHECKPOINT_PATH))
        training_epoch = int(config.GENERAL.CHECKPOINT_PATH.split('-')[-1].split('.')[0]) + 1
        print("TRAINING_EPOCH: ", training_epoch)
        print("Checkpoint {} found".format(config.GENERAL.CHECKPOINT_PATH))

    except:
        print("Checkpoint {} not found".format(config.GENERAL.CHECKPOINT_PATH))
        return

    model = model.to(device)
    model.eval()

    seg_label_to_cat = test_dataset.label_to_names
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: ", pytorch_total_params)
    print("Total trainable params: ", pytorch_total_trainable_params)
    
    print("\nEvaluating on {} samples".format(len(test_dataloader)))
    config.GENERAL.LEVEL = "FULL"
    with torch.no_grad():
        seg_evaluator = iouEval(config.SEGMENTATION.NUM_CLASSES, [])

        # for i, batch in enumerate(val_dataloader):
        for i, batch in enumerate(tqdm(test_dataloader)):
            _, complet_inputs, _, _ = batch
            
            try:
                results = model.inference(complet_inputs)
            except Exception as e:
                print(e, "Error in inference: ", i)
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
                print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=seg_label_to_cat[i], jacc=jacc*100))
        print('\nEval point avg class IoU: %f \n' % (m_jaccard*100))
    torch.cuda.empty_cache()

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="Reinforcement Learning - based path planning")
    parser.add_argument("--config-file", type=str, default="configs/ssc_eval.yaml", required=False)
    parser.add_argument("--output-path", type=str, default="experiments", required=False)

    args = parser.parse_args()
    config.GENERAL.OUT_DIR = args.output_path
    print("\n Training with configuration parameters: \n",config)
    config.merge_from_file(args.config_file)

    main()