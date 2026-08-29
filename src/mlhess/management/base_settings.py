import os
from pathlib import Path

import torch


def set_num_threads():
    # torch, sklearn, and tblite each bundle their own libomp.dylib on macOS,
    # which crashes with "OMP: Error #15" when more than one gets loaded.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads:
        return int(omp_threads)
    print("OMP_NUM_THREADS is not set. Falling back to one thread.")
    return 1


global NUM_THREADS #noQA: PLW0604
NUM_THREADS: int = set_num_threads()

global DEVICE #noQA: PLW0604
DEVICE: str = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    torch.cuda.set_device(DEVICE)

global PROCESSED_DATA_FOLDER #noQA: PLW0604
PROCESSED_DATA_FOLDER: str = "processed_data"

global PACKAGE_DIR #noQA: PLW0604
PACKAGE_DIR = os.path.abspath(Path(__file__).resolve().parent.parent)

