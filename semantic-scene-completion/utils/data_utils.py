import torch
from semantic_kitti_dataset import SemanticKITTIDataset, Merge, MergeTest
import numpy as np
import time

def get_dataloaders(config):
    # SemanticKITTI dataset
    train_dataset = SemanticKITTIDataset("train",do_overfit=config.GENERAL.OVERFIT, num_samples_overfit=config.GENERAL.NUM_SAMPLES_OVERFIT, augment=config.TRAIN.AUGMENT)
    if config.GENERAL.OVERFIT:
        val_dataset = SemanticKITTIDataset("train",do_overfit=True, num_samples_overfit=config.GENERAL.NUM_SAMPLES_OVERFIT)
        trainval_dataset = SemanticKITTIDataset("train",do_overfit=True, num_samples_overfit=config.GENERAL.NUM_SAMPLES_OVERFIT)
    else:
        val_dataset = SemanticKITTIDataset("valid",do_overfit=False)
        trainval_dataset = SemanticKITTIDataset("trainval",do_overfit=False)
    

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
    return train_dataloader, val_dataloader, trainval_dataloader

def get_test_dataloader(config):
    test_dataset = SemanticKITTIDataset("test",do_overfit=False, augment=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=MergeTest,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )
    return test_dataloader