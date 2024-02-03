import os
import psutil
import json
import metrics.globals

from pathlib import Path
from datetime import datetime

RESULT_PATH = "/data/result/"
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
# {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}


def has_image_extension(image_path):
    return image_path.lower().endswith(("png", "jpg", "jpeg", "webp"))


def check_file_exists(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)


def update_status(message, scope="metircs.CORE"):
    print(f"[{scope}] {message}")


def get_image_paths(path):
    path = Path(path)
    return sorted(
        [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
    )


def update_progress(progress, num=1):
    for i in range(num):
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
        progress.set_postfix(
            {
                "memory_usage": "{:.2f}".format(memory_usage).zfill(5) + "GB",
                "execution_threads": metrics.globals.execution_threads,
            }
        )
        progress.refresh()
        progress.update()


def set_global_result_path():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    result_path = f"{RESULT_PATH}{formatted_time}.json"
    metrics.globals.result_path = result_path
    return result_path


def save_result(key, value):
    file_path = metrics.globals.result_path
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    data[key] = value

    # 변경된 내용을 파일에 다시 쓰기
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
