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

            return rgbd_image_transformed

        return rgbd_image


class PointCloudTransform:
    """
    A class to apply transformations to point cloud data.

    Args:
    - rotation_angle (float): Maximum rotation angle for random rotation.
    - scale_range (tuple): Min and max scaling factor for random scaling.
    - translation_range (tuple): Min and max translation for random translation.
    - jitter (bool): Whether to apply random jitter to the points.

    Methods:
    - __call__: Applies the transformations to the point cloud data.
    """

    def __init__(
        self,
        rotation_angle=30,
        scale_range=(0.9, 1.1),
        translation_range=(-0.2, 0.2),
        jitter=True,
    ):
        self.rotation_angle = rotation_angle
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.jitter = jitter

    def __call__(self, point_cloud):

        # Random rotation
        angle = np.deg2rad(random.uniform(-self.rotation_angle, self.rotation_angle))
        rotation_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        point_cloud[:, :3] = np.dot(point_cloud[:, :3], rotation_matrix.T)

        # Random scaling
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        point_cloud[:, :3] *= scale

        # Random translation
        translation = np.random.uniform(
            self.translation_range[0], self.translation_range[1], size=3
        )
        point_cloud[:, :3] += translation

        # Random jitter
        if self.jitter:
            point_cloud[:, :3] += np.random.normal(
                0, 0.02, size=point_cloud[:, :3].shape
            )

        return point_cloud


# Define the augmentations to be applied to RGBD images
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

# Define the augmentations to be applied to point clouds
point_cloud_transform = PointCloudTransform()
