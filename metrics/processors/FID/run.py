import torch, threading
import metrics.globals

from metrics.utilites import update_status, check_file_exists, save_result
from .model import InceptionV3
from .service import compute_statistics_of_path, calculate_frechet_distance


NAME = "metircs.FID"
INCEPTION = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()


def get_inception(device, dims):
    global INCEPTION

    with THREAD_LOCK:
        if INCEPTION is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            model = InceptionV3([block_idx]).to(device)
            INCEPTION = model

    return INCEPTION


def clear_process():
    global INCEPTION
    INCEPTION = None


def pre_start():
    if not check_file_exists(metrics.globals.inception):
        update_status("Model Path is not Exsit", NAME)
    return True


def calculate_fid_given_paths(
    target_paths, swapped_paths, batch_size, dims, num_workers=1
):
    """Calculates the FID of two paths"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_inception(device, dims)

    m1, s1 = compute_statistics_of_path(
        target_paths, model, batch_size, dims, device, num_workers
    )
    m2, s2 = compute_statistics_of_path(
        swapped_paths, model, batch_size, dims, device, num_workers
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def start_process(_, target_paths, swapped_paths):
    with THREAD_SEMAPHORE:
        fid_score = calculate_fid_given_paths(
            target_paths,
            swapped_paths,
            metrics.globals.inception_batch_size,
            metrics.globals.inception_dims,
            metrics.globals.execution_threads,
        )
    save_result("FID_SCORE", fid_score)
