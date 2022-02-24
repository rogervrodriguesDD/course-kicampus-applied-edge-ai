import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import sys

def plot_dataset_sample_images(dataset, cols, rows, dir_fig, savefig=False):

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

    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        for key, value in logged_metrics.items():
            writer.writerow([key, value])

    return None

def load_logged_metrics(log_dir):
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
        axes[i].grid()
        axes[i].set_title(key)
        i += 1

    if plotfig:
        plt.show(block=True)

    if savefig:
        fig.savefig(dir_fig, format='png')

    return fig,  axes, colors
