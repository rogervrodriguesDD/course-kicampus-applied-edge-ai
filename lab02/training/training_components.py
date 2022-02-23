from collections import defaultdict
import statistics
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from tqdm import tqdm, trange
from typing import Union, Type

from image_classification.models import get_device, to_device

# Train process
loss_function = nn.CrossEntropyLoss()

# Optimizer
def get_optimizer(network: Type[nn.Module], learning_rate: float):
    return torch.optim.Adam(network.parameters(), lr=learning_rate)

def train_for_one_iteration(network: Type[nn.Module], batch: tuple, optimizer: Type[Optimizer]) -> float:

    images, labels = batch
    outputs = network(images)
    loss = loss_function(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())

def train(train_data: DataLoader, test_data: DataLoader, network: Type[nn.Module],
        optimizer: Type[Optimizer], num_epochs: int) -> dict:
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

        correct_predictions = (predictions_classes == labels).sum()

        num_correct_predictions += correct_predictions

    accuracy = num_correct_predictions / len(data_loader.dataset)
    return float(accuracy.item())
