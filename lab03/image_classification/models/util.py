import torch

# Getting Device / To Device
def get_device() -> torch.device:
    """Check if there is an GPU available"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Send data to the GPU"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)
