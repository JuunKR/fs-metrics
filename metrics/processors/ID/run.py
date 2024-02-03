import torch
import threading

from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import metrics.globals
from metrics.utilites import (
    update_status,
    check_file_exists,
    update_progress,
    save_result,
)
from metrics.multiprocess import multi_process
from .model import sphere

NAME = "metircs.ID"
COSFACE = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()


def get_cosface(device):
    global COSFACE

    with THREAD_LOCK:
        if COSFACE is None:
            model = sphere()
            model.load_state_dict(torch.load(metrics.globals.cosface))
            model.eval()
            model.to(device)
            COSFACE = model

    return COSFACE


def clear_process():
    global COSFACE
    COSFACE = None


def pre_start():
    if not check_file_exists(metrics.globals.cosface):
        update_status("Model Path is not Exsit", NAME)
    return True


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


def cosface(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_cosface(device)
    transform = transforms.Compose(
        [
            transforms.Resize((112, 92)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature_vector = model(input_tensor)
    return feature_vector


def calc_sim_score(source_path, swapped_path):
    with THREAD_SEMAPHORE:
        feature_vector1 = cosface(source_path)
        feature_vector2 = cosface(swapped_path)
        similarity_score = (
            cosine_sim(feature_vector1, feature_vector2).cpu().numpy()[0][0]
        )
        return similarity_score


def calc_sim_scores(source_paths, swapped_paths, update=None):
    total_score = None
    for source_path, swapped_path in zip(source_paths, swapped_paths):
        score = calc_sim_score(source_path, swapped_path)
        if not total_score:
            total_score = score
        else:
            total_score += score
    if update:
        update(len(source_paths))
    return total_score


def start_process(source_paths, _, swapped_paths):
    progress_bar_format = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    total = len(swapped_paths)
    with tqdm(
        total=total,
        desc="Processing",
        unit="frame",
        dynamic_ncols=True,
        bar_format=progress_bar_format,
    ) as progress:

        def call_back_func(num):
            update_progress(progress, num)

        total_scores = multi_process(
            source_paths, swapped_paths, calc_sim_scores, call_back_func
        )
    avg_score = sum(total_scores) / total
    save_result("ID_SCORE", avg_score)
