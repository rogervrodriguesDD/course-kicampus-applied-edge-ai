import matplotlib.pyplot as plt

from collections import defaultdict
from imgaug import augmenters as iaa
from pathlib import Path
import pickle
import statistics
from tqdm import tqdm, trange
from typing import Union, Type

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt

import util_scripts.cifar100_resnets as models

# Just to visualize some images
labels = pickle.load(Path('./data/cifar-100-python/meta').open('rb'))

def plot_img_examples():
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 4, 4
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        img_t = torch.permute(img, (1, 2, 0)) # Reordering to format HWC
        img_aug = train_augmentations(img_t.numpy())
        figure.add_subplot(rows, cols, i)
        plt.title(labels['fine_label_names'][label])
        plt.axis("off")
        plt.imshow(img_aug)
    figure.suptitle('Dataset examples (Normalized and Augmented)')
    plt.show(block=False)
    input('> Press enter to close the figure.')

# Getting Device / To Device
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)

# Defining the Neural Network
class CIFAR100Net(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.feature_extractor = models.resnet20(num_classes=num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor.forward(images)

# Train process
loss_function = nn.CrossEntropyLoss()

def train_for_one_iteration(network: Type[nn.Module], batch: tuple, optimizer: Type[Optimizer]) -> float:

    images, labels = batch
    if train_augmentations is not None:
        images = train_augmentations(torch.permute(images, (0, 2, 3, 1)).numpy())
    outputs = network(images)
    loss = loss_function(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())

def train(train_data: DataLoader, test_data: DataLoader, network: Type[nn.Module], optimizer: Type[Optimizer], num_epochs: int) -> dict:
    device = get_device()
    metrics = defaultdict(list)
    for epoch in trange(num_epochs, desc="Epoch: "):
        losses = []
        with tqdm(total=len(train_data), desc="Iteration: ") as progress_bar:
            for iteration, batch in enumerate(train_data):
                batch = to_device(batch, device)
                loss = train_for_one_iteration(network, batch, optimizer)
                losses.append(loss)
                progress_bar.update()
                metrics["losses"].append({"iteration": epoch * len(train_data) + iteration, "value": loss})

            accuracy = test_model(network, test_data)
            metrics['accuracy'].append({"iteration": (epoch + 1) * (len(train_data)), "value": accuracy})

            progress_bar.set_postfix_str(f"Epoch {epoch}, Mean Loss: {statistics.mean(losses):.2f}, Test Accuracy: {accuracy:.2f}")

    return metrics

# Network testing
def test_model(network: Type[nn.Module], data_loader: DataLoader) -> float:
    num_correct_predictions = 0
    device = get_device()

    for images, labels in tqdm(data_loader, desc='Testing...', leave=False):
        images = to_device(images, device)
        labels = to_device(labels, device)

        predictions = network(images)
        predictions_softmax = nn.functional.softmax(predictions, dim=1)
        _, predictions_classes = torch.max(predictions_softmax.data, dim=1)

        correct_predictions = (predicted_classes == labels).sum()

        num_correct_predictions += correct_predictions

    accuracy = num_correct_predictions / len(data_loader.dataset)
    return float(accuracy.item())

def main():

    # Loading the datasets
    image_transformation = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025)),
    ])

    train_data = CIFAR100(root='./data', train=True, transform=image_transformation, target_transform=None, download=True)
    test_data = CIFAR100(root='./data', train=False, transform=image_transformation, target_transform=None, download=True)

    # Defining train_augmentations
    train_augmentations = tt.Compose([
        iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.CropAndPad(percent=(-0.25, 0.25))
        ]).augment_images,
        tt.ToTensor()
    ])
    train_augmentations = None

    # plot_img_examples()

    # DataLoaders objects
    train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

    # Executing the training
    # Setting hyperparameters
    learning_rate = 0.001
    num_epochs = 50

    # Creating the Network
    network = CIFAR100Net()
    network = network.to(get_device())

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    # Training
    logged_metrics = train(train_data_loader, test_data_loader, network, optimizer, num_epochs)

if __name__ == '__main__':
    main()
