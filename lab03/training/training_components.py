from collections import defaultdict
import statistics
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from tqdm import tqdm, trange
from typing import Union, Type, List

from image_classification.models import get_device, to_device
from .losses_functions import get_teacher_loss_function, get_student_loss_function

# Losses functions
teacher_loss_function = get_teacher_loss_function()
student_loss_function = get_student_loss_function()

# Optimizer
def get_optimizer(network: Type[nn.Module], learning_rate: float):
    """
    Returns the implemented Adam optimizer with given Learning Rate.
    The values for betas and weight decay are default.

    Args:
        network (nn.Module): Network whose weights will be optimzed.
        learning_rate (float): Learning rate.

    Returns:
        Adam optimizer with lr=learning_rate.
    """
    return torch.optim.Adam(network.parameters(), lr=learning_rate)

def accuracy(predictions: torch.Tensor, labels: torch.Tensor, reduce_mean: bool = True) -> torch.Tensor:
    """
    Calculate accuracy for a given set of predictions and labels.

    Args:
        predictions (torch.Tensor): Tensor with predicted values
        labels (torch.Tensor): Tensor with expected, or real, values
        reduce_mean (bool): If True, returns the accuracy in percentage

    Returns:
        accuracy (torch.Tensor): Accuracy metric (number of correct predictions)
    """
    predicted_classes = torch.argmax(nn.functional.softmax(predictions, dim=1), dim=1)
    correct_predictions = torch.sum(predicted_classes == labels)
    if reduce_mean:
        return correct_predictions / len(labels)
    return correct_predictions

def train_for_one_iteration(networks: List[Type[nn.Module]], batch: tuple, optimizers: List[Type[Optimizer]]) -> dict:
    """
    Run a training step.

    Args:
        network (nn.Module): Network whose weights will be optimzed.
        batch (tuple): A batch with images and  labels to be used in the training step.
        optimizer (torch.Optimizer): Optimizer used for the network weights update.

    Returns:
        loss (float): Loss based on the predictions and labels of the batch

    """

    images, labels = batch
    teacher_network, student_network = networks

    # Forward pass and loss calculation for teacher and student networks
    teacher_predictions = teacher_network(images)
    teacher_loss = teacher_loss_function(teacher_predictions, labels)

    student_predictions = student_network(images)
    student_ce_loss = teacher_loss_function(student_predictions, labels)

    softmax_function = nn.Softmax(dim=1)
    sfmx_student_predictions = softmax_function(student_predictions)
    sfmx_teacher_predictions = softmax_function(teacher_predictions).detach() # disables backpropagation

    student_kd_loss = student_loss_function(sfmx_student_predictions, sfmx_teacher_predictions)
    student_loss = student_ce_loss + student_kd_loss

    # Accuracy for both predictions
    teacher_accuracy = accuracy(teacher_predictions, labels)
    student_accuracy = accuracy(student_predictions, labels)

    for loss, optimizer in zip([teacher_loss, student_loss], optimizers):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        "teacher_loss": float(teacher_loss.item()),
        "teacher_train_acc": float(teacher_accuracy.item()),
        "student_loss": float(student_loss.item()),
        "student_train_acc": float(student_accuracy.item()),
    }

def train(train_data: DataLoader, test_data: DataLoader, networks: List[Type[nn.Module]],
            optimizers: List[Type[Optimizer]], num_epochs: int) -> dict:
    """
    Run a training loop.

    Args:
        train_data (DataLoader): DataLoader with data for training.
        test_date (DataLoader): DataLoader with data for testing (calculation of accuracy).
        networks (list of nn.Module): Networks (teacher and student) whose weights will be optimized.
        optimizers (torch.Optimizer): Optimizers (teacher and student) used for the network weights update.
        num_epochs (int): Number of epochs executed during the training loop.

    Returns:
        metrics (dict): Dictionary with the metrics loss and accuracy as keys. Their values
                        are a list of dictionaries whose items are the iteraction number and the
                        values of respective metrics.
    """
    device = get_device()
    metrics = defaultdict(list)

    for epoch in trange(num_epochs, desc="Epoch: "):
        losses = defaultdict(list) 
        with tqdm(total=len(train_data), desc="Iteration: ") as progress_bar:
            for iteration, batch in enumerate(train_data):
                current_iteration = epoch * len(train_data) + iteration

                batch = to_device(batch, device)
                calculated_losses = train_for_one_iteration(networks, batch, optimizers)

                for loss_name, loss_value in calculated_losses.items():
                    losses[loss_name].append(loss_value)
                    metrics[loss_name].append({"iteration": current_iteration, "value": loss_value})

                postfix_data = {name: f"{value:.2f}" for name, value in calculated_losses.items()}
                progress_bar.set_postfix(postfix_data)
                progress_bar.update()

            progress_bar.set_description_str("Testing: ")
            accuracies = {}
            for metric_name, network in zip(["teacher_acc", "student_acc"], networks):
                accuracy = test_model(network, test_data)
                accuracies[f"{metric_name}"] = f"{accuracy:.2f}"
                metrics[metric_name].append({"iteration": (epoch + 1) * len(train_data), "value": accuracy})

            progress_bar.set_description_str(f"Epoch: {epoch}")
            postfix_data = {name: f"{statistics.mean(loss):.2f}" for name, loss in losses.items()}
            postfix_data.update()
            postfix_data.update(accuracies)
            progress_bar.set_postfix(postfix_data)
            progress_bar.update()

    return metrics

# Network testing
def test_model(network: Type[nn.Module], data_loader: DataLoader) -> float:
    """
    Obtain the predictions for the model, and based on them, calculate the
    its accuracy.

    Args:
        network (nn.Module): Network used for the test.
        data_loader (DataLoader): DataLoader with data for the test.

    Returns:
        accuracy (float): Accuracy based on predictions and labels.
    """
    num_correct_predictions = 0
    device = get_device()

    for images, labels in tqdm(data_loader, desc='Testing...', leave=False):
        images = to_device(images, device)
        labels = to_device(labels, device)

        predictions = network(images)
        num_correct_predictions += float(accuracy(predictions, labels, reduce_mean=False).item())

    return num_correct_predictions / len(data_loader.dataset)
