
def re_seed(seed: int = 0) -> None:
    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    from torch.backends import cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False  # not too much slower

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

def update_level(config,epoch):
    if epoch>=config.TRAIN.STEPS[0]:
        if epoch>=config.TRAIN.STEPS[1]:
            if epoch>=config.TRAIN.STEPS[2]:
                config.GENERAL.LEVEL = "FULL"
                config.TRAIN.EVAL_PERIOD = 1 if epoch >= (config.TRAIN.STEPS[2]+1) else 1
            else:
                config.GENERAL.LEVEL = "256"
        else:
            config.GENERAL.LEVEL = "128"