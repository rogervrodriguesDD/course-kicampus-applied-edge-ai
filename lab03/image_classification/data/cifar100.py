from imgaug import augmenters as iaa
from pathlib import Path
import pickle
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100 as cifar100_dataset
import torchvision.transforms as tt
from typing import Union, Type

class CIFAR100(cifar100_dataset):
    """
    CIFAR 100 DataModule.
    CIFAR100 DataModule from torchvision.datasets used as Parent Class.

    If the 'transform' argument is None, the default transformation pipeline is used (convert to tensor,
    followed by a Normalization of this tensor).

    An new parameter and method is defined for getting the fine_label_names of this Dataset.
    To get those labels, use the method `get_fine_labels_names`.    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fine_label_names = pickle.load(Path.joinpath(self.root,'./cifar-100-python/meta').resolve().open('rb'))['fine_label_names']

        if 'image_augmentations' in kwargs.keys():
            self.image_augmentations = kwargs['image_augmentations']
        else:
            self.image_augmentations = None

        if 'transform' in kwargs.keys():
            self.transform = kwargs['transform']
        else:
            self.transform = tt.Compose([ tt.ToTensor(), tt.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))])

    def get_fine_labels_names(self) -> list:
        return self.fine_label_names

    def is_augmented(self):
        return self.image_augmentations is not None
