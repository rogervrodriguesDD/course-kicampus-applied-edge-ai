import torch
import torch.nn as nn
from .cifar100_resnets import resnet20

class RESNET20(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.feature_extractor = resnet20(num_classes=num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor.forward(images)
