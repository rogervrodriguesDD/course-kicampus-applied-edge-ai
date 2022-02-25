"""Experiment-running."""

import argparse
import csv
from pathlib import Path
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
#import torchvision.transforms as tt
#from imgaug import augmenters as iaa
from torch.utils.data import DataLoader

from image_classification.data import CIFAR100, save_logged_metrics
from image_classification.models import RESNET20, to_device, get_device
from training.training_components import train, get_optimizer

def _setup_parser():
    """Setup Python's ArgumentParser with learning rate, num of epochs, and files directories"""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--help', '-h', action='help')

    # Directories for data and logs
    DOWNLOADED_DATA_DIR = '../datasets/cifar100/downloaded'
    LOG_DIR = './logs/w2_logged_metrics.csv'
    parser.add_argument('--data_dir', type=str, default=DOWNLOADED_DATA_DIR)
    parser.add_argument('--log_dir', type=str, default=LOG_DIR)

    # Training parameters
    learning_rate = 0.001
    num_epochs = 50
    parser.add_argument('--learning_rate', type=float, default=learning_rate)
    parser.add_argument('--num_epochs', type=int, default=num_epochs)

    return parser

def main():
    """
    Run an experiment.

    Example of command
    ```
    python run_experiment.py --learning_rate=0.001 --num_epochs=50 --data_dir=../datasets/cifar100/downloaded --log_dir=./logs/w2_logged_metrics.csv
    ```
    """

    parser = _setup_parser()
    args = parser.parse_args()

    # Loading the datasets
    DOWNLOADED_DATA_DIR = Path(args.data_dir).resolve()
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

    # DataLoaders objects
    train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

    # Executing the training
    # Setting hyperparameters
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs

    # Creating the Network
    network = RESNET20()
    network = network.to(get_device())

    # Optimizer
    optimizer = get_optimizer(network, learning_rate)

    # Training
    logged_metrics = train(train_data_loader, test_data_loader, network, optimizer, num_epochs)
    LOGS_DIR = Path(args.log_dir).resolve()
    save_logged_metrics(LOGS_DIR, logged_metrics)

if __name__ == '__main__':
    main()
