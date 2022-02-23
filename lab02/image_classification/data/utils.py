import matplotlib.pyplot as plt
import torch

def plot_dataset_sample_images(dataset, cols, rows):

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
