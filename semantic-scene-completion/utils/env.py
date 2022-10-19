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