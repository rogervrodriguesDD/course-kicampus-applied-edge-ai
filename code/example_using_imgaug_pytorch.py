from imgaug import augmenters as iaa
import torch
import torchvision.transforms as tt
import numpy as np

image = np.zeros((32,  32, 3), dtype=np.uint8)
image_pyt = torch.zeros((32, 32, 3))

print(type(image))
print(type(image_pyt))
print(type(image_pyt.numpy()))

# Defining train_augmentations
train_augmentations = iaa.Sequential([
    iaa.flip.Fliplr(0.5),
    iaa.size.CropAndPad(percent=(-0.25, 0.25))
])

# Wrapping train_augumentations using tt.Compose
image_transformation = tt.Compose([
    iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.CropAndPad(percent=(-0.25, 0.25))
    ]).augment_images,
    tt.ToTensor()
])

imag_aug = train_augmentations.augment_image(image)
imag_pyt_aug = image_transformation(image_pyt.numpy())
