import torch

def available_device() -> str:
    if not torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"