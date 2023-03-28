import os
from yacs.config import CfgNode as Node

_C = Node()

# -----------------------------------------------------------------------------
# GENERAL CONFIGS
# -----------------------------------------------------------------------------
_C.GENERAL = Node()
_C.GENERAL.TASK = "train"
_C.GENERAL.MANUAL_SEED = 123
_C.GENERAL.DATASET_DIR = "/usr/src/app/data/"
_C.GENERAL.OVERFIT = True
_C.GENERAL.NUM_SAMPLES_OVERFIT = 20
_C.GENERAL.LEVEL = "64"
_C.GENERAL.OUT_DIR = "experiments"
_C.GENERAL.CHECKPOINT_PATH = None #"/usr/src/app/semantic-scene-completion/experiments/56/modelFULL-35.pth"


# -----------------------------------------------------------------------------
# TRAIN CONFIGS
# -----------------------------------------------------------------------------
_C.TRAIN = Node()
_C.TRAIN.MAX_EPOCHS = 301
# _C.TRAIN.MAX_EPOCHS = 1500
_C.TRAIN.T = 123
_C.TRAIN.SEG_NUM_PER_CLASS = [55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858, 240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114, 9833174, 129609852, 4506626, 1168181]
_C.TRAIN.COMPLT_NUM_PER_CLASS = [7632350044, 15783539,  125136, 118809, 646799, 821951, 262978, 283696, 204750, 61688703, 4502961, 44883650, 2269923, 56840218, 15719652, 158442623, 2061623, 36970522, 1151988, 334146]
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.CHECKPOINT_PERIOD = 150
_C.TRAIN.EVAL_PERIOD = 1
# _C.TRAIN.STEPS = [5, 10, 15]
_C.TRAIN.STEPS = [30,60,70]
# _C.TRAIN.STEPS = [50, 125, 150]

# DATA AUGMENTATION
_C.TRAIN.AUGMENT = True
_C.TRAIN.NOISE_LEVEL = 0.01
# _C.TRAIN.FLIP_AUGMENT = True # unused
# Rotation
_C.TRAIN.ROT_AUG_Z = [-20., 20.]
_C.TRAIN.ROT_AUG_Y = [-2.,2.0]
_C.TRAIN.ROT_AUG_X = [-2.,2.0]
_C.TRAIN.RANDOM_PC_DROP_AUG = 0.1
_C.TRAIN.RANDOM_TRANSLATION_PROB = 0.1

_C.TRAIN.DROPOUT = 0.0 # probability of element being zeroed
_C.TRAIN.UNCERTAINTY_LOSS = True

# _C.TRAIN.STEPS = [15, 40, 100] # OVERFIT


# -----------------------------------------------------------------------------
# COMPLETION CONFIGS
# -----------------------------------------------------------------------------
_C.COMPLETION = Node()
_C.COMPLETION.VOXEL_SIZE = 0.2
_C.COMPLETION.POINT_CLOUD_RANGE = [0, -25.6, -2, 51.2, 25.6, 4.4]
_C.COMPLETION.FULL_SCALE = [256, 256, 32]
_C.COMPLETION.SECOND_SCALE = [128, 128, 16]
_C.COMPLETION.THIRD_SCALE = [64, 64, 8]
_C.COMPLETION.LOVASZ_LOSS_LAMBDA =0.001
  
# -----------------------------------------------------------------------------
# SEGMENTATION CONFIGS
# -----------------------------------------------------------------------------
_C.SEGMENTATION = Node() 
_C.SEGMENTATION.SCALE = 10   # VOXEL_SIZE = 1 / SCALE, SCALE 10 (1CM)
_C.SEGMENTATION.FULL_SCALE = [0, 2048]
_C.SEGMENTATION.USE_COORDS = False
_C.SEGMENTATION.NUM_CLASSES = 20
_C.SEGMENTATION.CHECKPOINT = None
_C.SEGMENTATION.TRAIN = False
_C.SEGMENTATION.SOFTMAX = False
_C.SEGMENTATION.SEG_MODEL = "2DPASS" # "2DPASS" or "vanila"



# -----------------------------------------------------------------------------
# MODEL CONFIGS
# -----------------------------------------------------------------------------
_C.MODEL = Node() 
_C.MODEL.OCCUPANCY_64_WEIGHT = 1.0
_C.MODEL.SEMANTIC_64_WEIGHT = 10.0 
_C.MODEL.DISC_64_WEIGHT = 0.0
_C.MODEL.GEN_64_WEIGHT = 0.0 #1e-6
_C.MODEL.SEMANTIC2D_64_WEIGHT = 0.0

_C.MODEL.OCCUPANCY_128_WEIGHT = 1.0
_C.MODEL.SEMANTIC_128_WEIGHT = 10.0   
_C.MODEL.OCCUPANCY_256_WEIGHT = 1.0
_C.MODEL.SEMANTIC_256_WEIGHT = 10.0   
_C.MODEL.PC_SEG_WEIGHT = 1.0   

_C.MODEL.NUM_OUTPUT_CHANNELS = 16
_C.MODEL.NUM_INPUT_FEATURES = 16
_C.MODEL.SEG_HEAD = True
_C.MODEL.UNET2D = False
_C.MODEL.USE_COORDS = True
_C.MODEL.COMPLETION_INTERACTION = True
_C.MODEL.DISTILLATION = True
_C.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
_C.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
_C.MODEL.BOTTOM_CROP = [480, 320]



# -----------------------------------------------------------------------------
# SOLVER CONFIGS
# -----------------------------------------------------------------------------
_C.SOLVER = Node() 
_C.SOLVER.BASE_LR = 0.004 # 0.01
_C.SOLVER.BETA_1 = 0.9 
_C.SOLVER.BETA_2 = 0.999   
_C.SOLVER.WEIGHT_DECAY = 0.0
# _C.SOLVER.DECAY_STEP = 40
_C.SOLVER.LR_DECAY_RATE = 0.999
_C.SOLVER.LR_CLIP = 1e-5


## CONSTANTS
LABEL_TO_NAMES = {0: 'unlabeled', 1: 'car', 2: 'bicycle', 3: 'motorcycle', 4: 'truck',
                               5: 'other-vehicle', 6: 'person', 7: 'bicyclist', 8: 'motorcyclist',
                               9: 'road', 10: 'parking', 11: 'sidewalk', 12: 'other-ground', 13: 'building',
                               14: 'fence', 15: 'vegetation', 16: 'trunk', 17: 'terrain', 18: 'pole',
                               19: 'traffic-sign', 20: 'other-object', 21: 'other-object'}
