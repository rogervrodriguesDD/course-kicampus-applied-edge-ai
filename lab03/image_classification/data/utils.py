import csv
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path
from typing import Type
import torch
import sys

def plot_dataset_sample_images(dataset, cols, rows, dir_fig, savefig=False):
    """
    Plot examples of images for the given dataset.
    When the argument savefig is set to True, a new figure is saved using
    the directory 'dir_fig'.
    """

    figure = plt.figure(figsize=(2*cols, 2*rows))
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        img_t = torch.permute(img, (1, 2, 0)) # Reordering to format HWC
        figure.add_subplot(rows, cols, i)
        plt.title(dataset.get_fine_labels_names()[label])
        plt.axis("off")
        plt.imshow(img_t)

    title = 'Dataset images samples'
    if dataset.is_augmented():
        title = title + ' (With augmentation transformation)'
    figure.suptitle(title)
    plt.show(block=True)

    if savefig:
        figure.savefig(dir_fig, format='png')

def save_logged_metrics(file_path, logged_metrics):
    """
    Save the variable logged_metrics as csv file, using the
    'file_path' as directory.
    """

    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        for key, value in logged_metrics.items():
            writer.writerow([key, value])

    return None

def load_logged_metrics(log_dir):
    """
    Load the file of logged metrics and return it as
    a dictionary.
    """
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

    metrics = {}
    with open(log_dir, 'r') as file:
        # reader = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            row_k_list = list(row.keys())
            row_v_list = list(row.values())
            for row_list in [row_k_list, row_v_list]:
                met_keys = row_list[0]
                met_values = row_list[1]
                metrics[met_keys] = json.loads(met_values.replace("'", '"'))

    return metrics

def plot_metrics(metrics: dict, dir_fig: Path, label: str, axes: np.array = None, colors: iter = None,
                plotfig:bool = True, savefig: bool = False):
    """
    Plot the metrics registered in the metrics dictionary.
    To make it possible to plot the values for different log files, it is possible to enter with the
    'axes' and 'colors' arguments, which make possible to plot on the same graph and with different color
    for each time the function is called.
    The arguments 'plotfig' and 'savefig' may then be adjusted to show and / or save the figure when it is needed.
    """
    fig = None
    if axes is None:
        fig, axes = plt.subplots(len(metrics), 1, sharex=True, figsize=(10, 5*len(metrics)))
    if colors is None:
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, 5)))

    color = next(colors)

    i = 0
    for key in metrics.keys():
        line_plot = np.array([(item['iteration'], item['value']) for item in metrics[key]])
        axes[i].plot(line_plot[:,0], line_plot[:,1], label=label, color=color)
        axes[i].legend()
        axes[i].set_title(key)
        axes[i].grid(visible=True)
        i += 1

    if plotfig:
        plt.show(block=True)

    if savefig:
        fig.savefig(dir_fig, format='png')

    return fig,  axes, colors

def plot_metrics_lab03(metrics: dict, dir_fig: Type[Path], label: str,
                    fig: Type[Figure] = None, axes: np.array = None,
                    colors: iter = None,
                    plotfig:bool = True, savefig: bool = False):
    """
    Plot the metrics registered in the metrics dictionary.
    To make it possible to plot the values for different log files, it is possible to enter with the
    'axes' and 'colors' arguments, which make possible to plot on the same graph and with different color
    for each time the function is called.
    The arguments 'plotfig' and 'savefig' may then be adjusted to show and / or save the figure when it is needed.
    """
    if axes is None:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 30))
    if colors is None:
        colors = iter(plt.cm.tab20(np.linspace(0, 1, 10)))

    color_teacher = next(colors)
    color_student = next(colors)

    axes_metrics = {'loss': 0, 'train_acc': 1, 'acc': 2, 'lr': 3}

    i = 0
    for key in metrics.keys():
        label_metric, actual_metric = key.split('_', 1)

        idx_plot = axes_metrics[actual_metric]
        if label_metric == 'teacher':
            linestyle = 'solid'
            color = color_teacher
        else:
            linestyle = 'dashed'
            color = color_student

        line_plot = np.array([(item['iteration'], item['value']) for item in metrics[key]])
        axes[idx_plot].plot(line_plot[:,0], line_plot[:,1], label="{} ({})".format(label, label_metric), color=color, linestyle=linestyle, alpha=0.7)
        axes[idx_plot].legend(bbox_to_anchor=(1, 1))
        axes[idx_plot].set_title(actual_metric)
        axes[idx_plot].grid(visible=True)
        i += 1

    if plotfig:
        plt.tight_layout(rect=[0,0.05,1.0,0.95], h_pad=5)
        plt.show(block=False)
        user_input = input('> Press enter to close the figure')
        plt.close()

    if savefig:
        fig.savefig(dir_fig, format='png')

    return fig,  axes, colors
