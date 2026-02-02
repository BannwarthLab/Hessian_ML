import os
import torch


def set_num_threads():
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads:
        return int(omp_threads)
    print("OMP_NUM_THREADS is not set. Falling back to one thread.")
    return 1


global NUM_THREADS
NUM_THREADS: int = set_num_threads()

global DEVICE
DEVICE: str = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    torch.cuda.set_device(DEVICE)

global PROCESSED_DATA_FOLDER
PROCESSED_DATA_FOLDER: str = "processed_data"
