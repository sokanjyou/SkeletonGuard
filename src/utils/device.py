from __future__ import annotations
from typing import Any
import torch


def resolve_torch_device(
    device: str | int | None = "auto",
    prefer_cuda: bool = True,
    cuda_index: int = 0,
) -> torch.device:
    """

    Resolve a PyTorch device with explicit CUDA preference. If CUDA is
    available and preferred, the device is resolved to a CUDA device with
    the specified index. ^v^
    
    """

    if device is None or str(device).lower() == "auto":
        if prefer_cuda and torch.cuda.is_available():
            return torch.device(f"cuda:{int(cuda_index)}")
        return torch.device("cpu")
    return torch.device(device)


def resolve_yolo_device(
    device: str | int | None = "auto",
    prefer_cuda: bool = True,
    cuda_index: int = 0,
) -> Any:
    """
    
    Resolve an Ultralytics device argument. ^v^


    """

    if device is None or str(device).lower() == "auto":
        if prefer_cuda and torch.cuda.is_available():
            return int(cuda_index)
        return "cpu"
    return device
