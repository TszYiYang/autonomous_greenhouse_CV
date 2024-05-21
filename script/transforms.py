"""
transforms.py

This module contains the data augmentation transformations to be applied to 4-channel RGBD images.

Functions:
- Transform: A class to apply the same transformations to 4-channel RGBD images.
"""

from torchvision import transforms
import random
import numpy as np
from PIL import Image


class Transform:
    """
    A class to apply the same transformations to 4-channel RGBD images.

    Args:
    - transform (callable): A function/transform to apply to the images.

    Methods:
    - __call__: Applies the transformation to the RGBD image.
    """

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, rgbd_image):
        if self.transform:
            # Convert the numpy array to a PIL image
            rgbd_image_pil = Image.fromarray(np.uint8(rgbd_image.transpose(1, 2, 0)))

            # Apply the transformation
            rgbd_image_transformed = self.transform(rgbd_image_pil)

            # Convert back to numpy array
            rgbd_image_transformed = np.array(rgbd_image_transformed)

            # Transpose to match PyTorch's expected format
            rgbd_image_transformed = rgbd_image_transformed.transpose(0, 1, 2)

            # print(f"Transformed RGBD image shape: {rgbd_image_transformed.shape}")

            return rgbd_image_transformed
        return rgbd_image


# Define the augmentations to be applied
# Define the augmentations to be applied
transform = Transform(
    transform=transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ToTensor(),
        ]
    )
)
