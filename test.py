import sys
import time
import torch
import random
sys.path.append("build")
import PYTORCH_GROUPED_GEMM

# for high-end gpu, it defaults to tf32 instead of fp32
from packaging import version
if version.parse(torch.__version__) >= version.parse("1.7"):
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = False

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == "__main__":
 