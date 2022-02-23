from pathlib import Path
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch

from image_classification.data import CIFAR100, plot_dataset_sample_images

# Getting the train dataset
DOWNLOADED_DATA_DIR = Path('./image_classification/data/downloaded').resolve()
train_data = CIFAR100(root = DOWNLOADED_DATA_DIR,
                    train = True,
                    transform = ToTensor(),
                    download=True)

print('Image shape (format CHW):', train_data[0][0].shape)

# Plotting same examples
plot_dataset_sample_images(train_data, cols=4, rows=4)
