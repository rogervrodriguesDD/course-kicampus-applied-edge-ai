import csv
from pathlib import Path
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
#import torchvision.transforms as tt
#from imgaug import augmenters as iaa
from torch.utils.data import DataLoader

from image_classification.data import CIFAR100
from image_classification.models import RESNET20, to_device, get_device
from training.training_components import train, get_optimizer

def main():

    # Loading the datasets
    DOWNLOADED_DATA_DIR = Path('./image_classification/data/downloaded').resolve()
    train_data = CIFAR100(root = DOWNLOADED_DATA_DIR, train = True, download=True)
    test_data = CIFAR100(root = DOWNLOADED_DATA_DIR, train = False, download=True)

    # Defining train_augmentations wrapped with tt.Compose
    #train_augmentations = tt.Compose([
    #    iaa.Sequential([
    #        iaa.Fliplr(0.5),
    #        iaa.CropAndPad(percent=(-0.25, 0.25))
    #    ]).augment_images,
    #    tt.ToTensor()
    #])

    train_augmentations = None

    # DataLoaders objects
    train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

    # Executing the training
    # Setting hyperparameters
    learning_rate = 0.001
    num_epochs = 1

    # Creating the Network
    network = RESNET20()
    network = network.to(get_device())

    # Optimizer
    optimizer = get_optimizer(network, learning_rate)

    # Training
    logged_metrics = train(train_data_loader, test_data_loader, network, optimizer, num_epochs)

    LOGS_DIR = Path('./logs/w2_logged_metrics.csv').resolve()
    with open(LOGS_DIR, 'w') as file:
        writer = csv.writer(file)
        for key, value in logged_metrics.items():
            writer.writerow([key, value])

if __name__ == '__main__':
    main()
