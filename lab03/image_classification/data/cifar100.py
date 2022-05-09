from imgaug import augmenters as iaa
from pathlib import Path
import pickle
from PIL import Image
from torch import permute
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

        self.image_augmentations = iaa.Sequential([
                                        iaa.Fliplr(0.5),
                                        iaa.CropAndPad(percent=(-0.25, 0.25))
                                    ])

        if 'transform' in kwargs.keys():
            self.transform = kwargs['transform']
        else:
            self.transform = tt.Compose([ tt.ToTensor(), tt.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))])

    def get_fine_labels_names(self) -> list:
        return self.fine_label_names

    def is_augmented(self):
        return self.image_augmentations is not None

    def __getitem__(self, index: int) -> tuple:
        """
        Overwriten version of the function __getitem__ to add the augmenters functions of the imgaug library.
        """
        img, target = self.data[index], self.targets[index]

        if self.image_augmentations is not None:
            img = self.image_augmentations.augment_image(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
