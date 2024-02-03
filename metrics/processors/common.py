import importlib
import sys
import traceback
import torch
import torch.nn.functional as F

FRAME_PROCESSORS_INTERFACE = ["pre_start", "clear_process", "start_process"]


def load_metric_module(metric):
    try:
        metric_module = importlib.import_module(f"metrics.processors.{metric}.run")
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(metric_module, method_name):
                raise NotImplementedError
    except ModuleNotFoundError:
        err = traceback.format_exc()
        print(err)
        sys.exit(f"processor {metric} not found.")
    except NotImplementedError:
        sys.exit(f"processor {metric} not implemented correctly.")
    return metric_module


def get_metric_modules(metrics):
    processor_moduls = []
    for metirc in metrics:
        metric_module = load_metric_module(metirc)
        processor_moduls.append(metric_module)
    return processor_moduls


def cals_euclidean_avg(tensor1, tensor2, dim=0):
    distances = torch.sqrt(torch.sum((tensor1 - tensor2) ** 2, dim))
    average_distance = torch.mean(distances)
    return average_distance


def cals_l2_loss(tensor1, tensor2):
    return F.mse_loss(tensor1, tensor2)
