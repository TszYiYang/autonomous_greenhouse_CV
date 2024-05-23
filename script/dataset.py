import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import open3d as o3d
from transforms import transform


def filter_json_data(json_path, rgb_folder, depth_folder):
    """
    Filters out data points with missing values in the ground truth JSON file.

    Args:
    - json_path (str): Path to the JSON file containing ground truth labels.
    - rgb_folder (str): Path to the folder containing RGB images.
    - depth_folder (str): Path to the folder containing depth images.

    Returns:
    - dict: Filtered data dictionary with no missing values.
    - list: List of filtered image IDs.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    filtered_data = {}
    filtered_image_ids = []

    for image_id, traits in data.items():
        if not any(
            value is None or (isinstance(value, float) and np.isnan(value))
            for value in traits.values()
        ):
            rgb_path = os.path.join(rgb_folder, f"{image_id}.png")
            depth_path = os.path.join(depth_folder, f"{image_id}_depth.png")
            if os.path.exists(rgb_path) and os.path.exists(depth_path):
                filtered_data[image_id] = traits
                filtered_image_ids.append(image_id)

    return filtered_data, filtered_image_ids


class TomatoDataset(Dataset):
    """
    Custom Dataset for loading tomato dataset.

    Args:
    - rgb_folder (str): Path to the folder containing RGB images.
    - depth_folder (str): Path to the folder containing depth images.
    - pcd_folder (str): Path to the folder containing point cloud data.
    - filtered_data (dict): Filtered ground truth data.
    - filtered_image_ids (list): List of filtered image IDs.
    - transform (callable, optional): A function/transform to apply to the images.

    Methods:
    - __len__: Returns the length of the dataset.
    - __getitem__: Returns the image and its corresponding label.
    """

    def __init__(
        self,
        rgb_folder,
        depth_folder,
        pcd_folder,
        filtered_data,
        filtered_image_ids,
        transform=None,
    ):
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.pcd_folder = pcd_folder
        self.filtered_data = filtered_data
        self.filtered_image_ids = filtered_image_ids
        self.transform = transform

    def __len__(self):
        return len(self.filtered_image_ids)

    def __getitem__(self, idx):
        image_id = self.filtered_image_ids[idx]
        rgb_path = os.path.join(self.rgb_folder, f"{image_id}.png")
        depth_path = os.path.join(self.depth_folder, f"{image_id}_depth.png")
        pcd_path = os.path.join(self.pcd_folder, f"{image_id}_pcd.pcd")

        # Load images using PIL
        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = Image.open(depth_path)

        rgb_array = np.array(rgb_image)
        depth_array = np.array(depth_image)

        # Convert images to PyTorch tensors and apply transformations
        rgb_image = np.transpose(rgb_array, (2, 0, 1))
        depth_image = np.expand_dims(depth_array, axis=0)  # Add channel dimension
        rgbd_image = np.concatenate((rgb_image, depth_image), axis=0)

        if self.transform:
            rgbd_image = self.transform(rgbd_image)

        rgbd_image = torch.tensor(rgbd_image, dtype=torch.float32)

        # Load point cloud from PCD file
        if not os.path.exists(pcd_path):
            raise FileNotFoundError(f"Point cloud file not found: {pcd_path}")
        pcd = o3d.io.read_point_cloud(pcd_path)
        point_cloud = np.asarray(pcd.points)
        point_cloud_colors = np.asarray(pcd.colors)
        point_cloud = np.concatenate((point_cloud, point_cloud_colors), axis=1)
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

        label = torch.tensor(
            list(self.filtered_data[image_id].values()), dtype=torch.float32
        )

        return rgbd_image, point_cloud, label
