import torch
import torchvision
import torch.nn.functional as F
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
from metrics.processors.common import cals_euclidean_avg
from .model import Hopenet


NAME = "metircs.POSE"
HOPENET = None

THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()


def get_hopenet(device):
    global HOPENET
    with THREAD_LOCK:
        if HOPENET is None:
            model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
            model.load_state_dict(torch.load(metrics.globals.hopenet))
            model.eval()
            model.to(device)
            HOPENET = model
    return HOPENET


def clear_process():
    global HOPENET
    HOPENET = None


def pre_start():
    if not check_file_exists(metrics.globals.hopenet):
        update_status("Model Path is not Exsit", NAME)
    return True


def get_pose(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    model = get_hopenet(device)
    yaw, pitch, roll = model(image)

    yaw_predicted = F.softmax(yaw, dim=1)
    pitch_predicted = F.softmax(pitch, dim=1)
    roll_predicted = F.softmax(roll, dim=1)

    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    return [yaw_predicted, pitch_predicted, roll_predicted]


def calc_pose_score(target_path, swapped_path):
    with THREAD_SEMAPHORE:
        pose1 = get_pose(target_path)
        pose1 = torch.tensor(pose1, dtype=torch.float32)

        pose2 = get_pose(swapped_path)
        pose2 = torch.tensor(pose2, dtype=torch.float32)

        euclidean_avg = cals_euclidean_avg(pose1, pose2).cpu().numpy()
        return euclidean_avg


def calc_pose_scores(target_paths, swapped_paths, update=None):
    total_score = None
    for target_path, swapped_path in zip(target_paths, swapped_paths):
        score = calc_pose_score(target_path, swapped_path)
        if not total_score:
            total_score = score
        else:
            total_score += score
    if update:
        update(len(target_paths))
    return total_score


def start_process(_, target_paths, swapped_paths):
    progress_bar_format = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    total = len(target_paths)
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
            target_paths, swapped_paths, calc_pose_scores, call_back_func
        )
    avg_score = sum(total_scores) / total
    save_result("POSE_SCORE", avg_score)
