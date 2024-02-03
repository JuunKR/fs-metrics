import metrics.globals
import torch, threading, dlib, cv2, sys

from tqdm import tqdm

from metrics.processors.common import cals_euclidean_avg
from metrics.utilites import (
    update_status,
    check_file_exists,
    update_progress,
    save_result,
)
from metrics.multiprocess import multi_process


NAME = "metircs.EXPRESSION"
DLIB_DETECTOR = None
DLIB_PREDICTOR = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()


def get_detector():
    global DLIB_DETECTOR

    with THREAD_LOCK:
        if DLIB_DETECTOR is None:
            DLIB_DETECTOR = dlib.get_frontal_face_detector()

    return DLIB_DETECTOR


def get_predictor():
    global DLIB_PREDICTOR

    with THREAD_LOCK:
        if DLIB_PREDICTOR is None:
            DLIB_PREDICTOR = dlib.shape_predictor(metrics.globals.dlib)

    return DLIB_PREDICTOR


def clear_process():
    global DLIB_DETECTOR
    global DLIB_PREDICTOR

    DLIB_DETECTOR = None
    DLIB_PREDICTOR = None


def pre_start():
    if not check_file_exists(metrics.globals.dlib):
        update_status("Model Path is not Exsit", NAME)
    return True


def detect_face(image):
    model = get_detector()
    faces = model(image)
    return faces


def extract_lmrk(image_path):
    image = cv2.imread(str(image_path))
    faces = detect_face(image)
    if len(faces) > 1:
        update_status("More than one face has been extracted.", NAME)
        sys.exit()
    model = get_predictor()
    lmrk = model(image, faces[0])
    lmrk = [[points.x, points.y] for points in lmrk.parts()]
    return lmrk


def calc_expression_score(target_path, swapped_path):
    with THREAD_SEMAPHORE:
        lmrk1 = extract_lmrk(target_path)
        lmrk1 = torch.tensor(lmrk1, dtype=torch.float32)

        lmrk2 = extract_lmrk(swapped_path)
        lmrk2 = torch.tensor(lmrk2, dtype=torch.float32)
        euclidean_avg = cals_euclidean_avg(lmrk1, lmrk2, dim=1).cpu().numpy()
        return euclidean_avg


def calc_expression_scores(target_paths, swapped_paths, update=None):
    total_score = None
    for target_path, swapped_path in zip(target_paths, swapped_paths):
        score = calc_expression_score(target_path, swapped_path)
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
            target_paths, swapped_paths, calc_expression_scores, call_back_func
        )
    avg_score = sum(total_scores) / total
    save_result("EXPRESSION_SCORE", avg_score)
