import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor

train_data = CIFAR100(root='data', train=True, transform=ToTensor(), target_transform=None, download=True)
labels = pickle.load(Path('data/cifar-100-python/meta').open('rb'))

print(train_data)
print('Image shape (format CHW):', train_data[0][0].shape)

figure = plt.figure(figsize=(8, 8))
cols, rows = 4, 4
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    img_t = torch.permute(img, (1, 2, 0)) # Reordering to format HWC
    figure.add_subplot(rows, cols, i)
    plt.title(labels['fine_label_names'][label])
    plt.axis("off")
    plt.imshow(img_t)
plt.show()
