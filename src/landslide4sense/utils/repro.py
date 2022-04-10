import os
import random

import torch
import numpy as np


def set_deterministic(seed: int):
    # Replaced use_deterministic_algorithms(True) by this 
    if torch.cuda.is_available():
        # https://discuss.pytorch.org/t/torch-deterministic-algorithms-error/125200/4
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # This is done to ensure deterministic behaviour when using GPU and avoid the following error:
        # RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
