import torch
import torch.backends.mps
from torch import nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def should_use_data_parallel(device) -> bool:
    return (
            torch.device(device).type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
    )


def with_data_parallel(model: nn.Module, device: torch.device) -> torch.nn.Module:
    model = model.to(device)

    if isinstance(model, nn.DataParallel):
        return model

    if should_use_data_parallel(device):
        print(f"Using nn.DataParallel on {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(model)

    return model

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model
