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
from image_classification.models import RESNET, to_device, get_device
from training.training_components import train, get_optimizer, get_lr_scheduler

def _setup_parser():
    """Setup Python's ArgumentParser with learning rate, num of epochs, and files directories"""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--help', '-h', action='help')

    # Directories for data and logs
    DOWNLOADED_DATA_DIR = '../datasets/cifar100/downloaded'
    LOG_DIR = './logs/w3_logged_metrics.csv'
    parser.add_argument('--log_dir', type=str, default=LOG_DIR)
    parser.add_argument('--data_dir', type=str, default=DOWNLOADED_DATA_DIR)

    # Training parameters
    learning_rate = 0.001
    num_epochs = 50
    batch_size=128
    parser.add_argument('--learning_rate', type=float, default=learning_rate)
    parser.add_argument('--num_epochs', type=int, default=num_epochs)
    parser.add_argument('--batch_size', type=int, default=batch_size)

    # Teacher and student models
    teacher_model_class = 'resnet56'
    teacher_model_temperature = 1
    student_model_class = 'resnet20'
    student_model_temperature = 1
    parser.add_argument('--teacher_model_class', type=str, default=teacher_model_class)
    parser.add_argument('--teacher_model_temperature', type=str, default=teacher_model_temperature)
    parser.add_argument('--student_model_class', type=str, default=student_model_class)
    parser.add_argument('--student_model_temperature', type=str, default=student_model_temperature)

    # Update lr_scheduler for each iteration
    parser.add_argument('--update_lr_sch_each_iter', type=bool, default=True)

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

    # Creating the Networks
    teacher_model = RESNET(model_type=args.teacher_model_class, num_classes=100, temperature=args.teacher_model_temperature)
    student_model = RESNET(model_type=args.student_model_class, num_classes=100, temperature=args.student_model_temperature)

    teacher_model = teacher_model.to(get_device())
    student_model = student_model.to(get_device())

    # Optimizers
    teacher_optimizer = get_optimizer(teacher_model, learning_rate=learning_rate)
    student_optimizer = get_optimizer(student_model, learning_rate=learning_rate)

    teacher_scheduler = get_lr_scheduler(teacher_optimizer, learning_rate, num_epochs, len(train_data_loader))
    student_scheduler = get_lr_scheduler(student_optimizer, learning_rate, num_epochs, len(train_data_loader))

    # Training
    logged_metrics = train(
                        train_data_loader,
                        test_data_loader,
                        [teacher_model, student_model],
                        [teacher_optimizer, student_optimizer],
                        [teacher_scheduler, student_scheduler],
                        num_epochs,
                        True) # update_lr_scheduler_each_iteration

    LOGS_DIR = Path(args.log_dir).resolve()
    save_logged_metrics(LOGS_DIR, logged_metrics)

if __name__ == '__main__':
    main()
