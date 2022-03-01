import torch
import torch.nn as nn
import image_classification.models.cifar100_resnets as models 

class RESNET(nn.Module):
    """
    Residual Net with configurable layers.
    The feature extraction for this network was pre-defined and implemented by Yerlan Idelbayev, and
    can be found here: https://www.kaggle.com/bartzi/cifar100-resnets.
    """
    def __init__(self, model_type: str = 'resnet18', num_classes: int = 100, temperature: int = 1):
        super().__init__()
        model_class = getattr(models, model_type)
        self.feature_extractor = model_class(num_classes=num_classes)
        self.temperature = temperature

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        activations = self.feature_extractor.forward(images)
        return activations / self.temperature
