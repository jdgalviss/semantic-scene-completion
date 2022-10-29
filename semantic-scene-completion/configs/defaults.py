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
_C.GENERAL.OVERFIT = False
_C.GENERAL.NUM_SAMPLES_OVERFIT = 100
_C.GENERAL.LEVEL = "64"
_C.GENERAL.OUT_DIR = "experiments"

# -----------------------------------------------------------------------------
# TRAIN CONFIGS
# -----------------------------------------------------------------------------
_C.TRAIN = Node()
_C.TRAIN.MAX_STEPS = 200000
# _C.TRAIN.MAX_STEPS = 50000
_C.TRAIN.T = 123
_C.TRAIN.SEG_NUM_PER_CLASS = [55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858, 240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114, 9833174, 129609852, 4506626, 1168181]
_C.TRAIN.COMPLT_NUM_PER_CLASS = [7632350044, 15783539,  125136, 118809, 646799, 821951, 262978, 283696, 204750, 61688703, 4502961, 44883650, 2269923, 56840218, 15719652, 158442623, 2061623, 36970522, 1151988, 334146]
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.NUM_WORKERS = 1
_C.TRAIN.CHECKPOINT_PERIOD = 2
_C.TRAIN.EVAL_PERIOD = 2
_C.TRAIN.STEPS = [10001, 25001, 50s001]
# _C.TRAIN.STEPS = [201, 501, 1201]



# -----------------------------------------------------------------------------
# COMPLETION CONFIGS
# -----------------------------------------------------------------------------
_C.COMPLETION = Node()
_C.COMPLETION.VOXEL_SIZE = 0.2
_C.COMPLETION.POINT_CLOUD_RANGE = [0, -25.6, -2, 51.2, 25.6, 4.4]
_C.COMPLETION.FULL_SCALE = [256, 256, 32]
  
# -----------------------------------------------------------------------------
# SEGMENTATION CONFIGS
# -----------------------------------------------------------------------------
_C.SEGMENTATION = Node() 
_C.SEGMENTATION.SCALE = 10   # VOXEL_SIZE = 1 / SCALE, SCALE 10 (1CM)
_C.SEGMENTATION.FULL_SCALE = [0, 2048]
_C.SEGMENTATION.USE_COORDS = False
_C.SEGMENTATION.NUM_CLASSES = 20

# -----------------------------------------------------------------------------
# MODEL CONFIGS
# -----------------------------------------------------------------------------
_C.MODEL = Node() 
_C.MODEL.OCCUPANCY_64_WEIGHT = 0.1
_C.MODEL.SEMANTIC_64_WEIGHT = 0.1  
_C.MODEL.OCCUPANCY_128_WEIGHT = 0.2
_C.MODEL.SEMANTIC_128_WEIGHT = 0.2   
_C.MODEL.OCCUPANCY_256_WEIGHT = 2.0
_C.MODEL.SEMANTIC_256_WEIGHT = 2.0   
_C.MODEL.NUM_OUTPUT_CHANNELS = 16
_C.MODEL.UNET_FEATURES = 16


# -----------------------------------------------------------------------------
# SOLVER CONFIGS
# -----------------------------------------------------------------------------
_C.SOLVER = Node() 
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BETA_1 = 0.9 
_C.SOLVER.BETA_2 = 0.999   
_C.SOLVER.WEIGHT_DECAY = 0.0  

