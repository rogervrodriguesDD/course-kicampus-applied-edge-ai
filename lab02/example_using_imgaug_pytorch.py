from imgaug import augmenters as iaa
import torch
import torchvision.transforms as tt
import numpy as np

images = np.zeros((64, 32,  32, 3), dtype=np.uint8)
images_pyt = torch.zeros((64, 32, 32, 3))

print(type(images))
print(type(images_pyt))
print(type(images_pyt.numpy()))

# Defining train_augmentations
train_augmentations = iaa.Sequential([
    iaa.flip.Fliplr(0.5),
    iaa.size.CropAndPad(percent=(-0.25, 0.25))
])

# Wrapping train_augumentations using tt.Compose
train_augmentations_comp = tt.Compose([
    iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.CropAndPad(percent=(-0.25, 0.25))
    ]).augment_images,
    tt.ToTensor()
])

def tt_compose_handler(iaa_trans_func):
    def wrapper(*args, **kwargs):
        return None
    return None

imag_aug = train_augmentations.augment_images(images)
print('Transformation iaa ok!')
imag_pyt_aug = train_augmentations_comp(images_pyt.numpy())
