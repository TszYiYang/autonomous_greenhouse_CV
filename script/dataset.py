import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import open3d as o3d
from transforms import transform, point_cloud_transform


def filter_json_data(json_path, rgb_folder, depth_folder, pcd_folder=None):
    """
    Filters out data points with missing values in the ground truth JSON file.

    Args:
    - json_path (str): Path to the JSON file containing ground truth labels.
    - rgb_folder (str): Path to the folder containing RGB images.
    - depth_folder (str): Path to the folder containing depth images.
    - pcd_folder (str, optional): Path to the folder containing point cloud data (if applicable).

    Returns:
    - dict: Filtered data dictionary with no missing values.
    - list: List of filtered image IDs.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    filtered_data = {}
    filtered_image_ids = []
    total_points = len(data)
    missing_value_points = 0

    for image_id, traits in data.items():
        if not any(
            value is None or (isinstance(value, float) and np.isnan(value))
            for value in traits.values()
        ):
            rgb_path = os.path.join(rgb_folder, f"{image_id}.png")
            depth_path = os.path.join(depth_folder, f"{image_id}_depth.png")
            if pcd_folder:
                pcd_path = os.path.join(pcd_folder, f"{image_id}_pcd.pcd")
                if (
                    os.path.exists(rgb_path)
                    and os.path.exists(depth_path)
                    and os.path.exists(pcd_path)
                ):
                    filtered_data[image_id] = traits
                    filtered_image_ids.append(image_id)
            else:
                if os.path.exists(rgb_path) and os.path.exists(depth_path):
                    filtered_data[image_id] = traits
                    filtered_image_ids.append(image_id)
        else:
            missing_value_points += 1

    print(f"Total data points: {total_points}")
    print(f"Data points with missing values filtered out: {missing_value_points}")
    print(f"Remaining data points: {len(filtered_data)}")

    return filtered_data, filtered_image_ids


class TomatoDataset(Dataset):
    """
    Custom Dataset for loading tomato dataset.

    Args:
    - rgb_folder (str): Path to the folder containing RGB images.
    - depth_folder (str): Path to the folder containing depth images.
    - json_data (dict): Filtered ground truth data.
    - image_ids (list): List of filtered image IDs.
    - transform (callable, optional): A function/transform to apply to the images.
    - point_cloud_transform (callable, optional): A function/transform to apply to the point cloud data.
    - pcd_folder (str or None): Path to the folder containing point cloud data (if applicable, only for combo models).

    Methods:
    - __len__: Returns the length of the dataset.
    - __getitem__: Returns the image and its corresponding label.
    """

    def __init__(
        self,
        rgb_folder,
        depth_folder,
        json_data,
        image_ids,
        transform=None,
        point_cloud_transform=None,
        pcd_folder=None,
    ):
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.json_data = json_data
        self.image_ids = image_ids
        self.transform = transform
        self.point_cloud_transform = point_cloud_transform
        self.pcd_folder = pcd_folder

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        rgb_path = os.path.join(self.rgb_folder, f"{image_id}.png")
        depth_path = os.path.join(self.depth_folder, f"{image_id}_depth.png")

        # Load images using PIL
        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = Image.open(depth_path)

        rgb_array = np.array(rgb_image)
        depth_array = np.array(depth_image)

        rgb_image = np.transpose(
            rgb_array, (2, 0, 1)
        )  # Convert to [channels, height, width]
        depth_image = np.expand_dims(
            depth_array, axis=0
        )  # Add channel dimension: [1, height, width]
        rgbd_image = np.concatenate(
            (rgb_image, depth_image), axis=0
        )  # Combine RGB and Depth: [4, height, width]

        if self.transform:
            rgbd_image = self.transform(rgbd_image)

        rgbd_image = torch.tensor(rgbd_image, dtype=torch.float32)
        # print(f"Processed RGBD image shape: {rgbd_image.shape}")

        # Load point cloud from PCD file if applicable
        if self.pcd_folder:
            pcd_path = os.path.join(self.pcd_folder, f"{image_id}_pcd.pcd")
            if not os.path.exists(pcd_path):
                raise FileNotFoundError(f"Point cloud file not found: {pcd_path}")
            pcd = o3d.io.read_point_cloud(pcd_path)
            point_cloud = np.asarray(pcd.points)
            point_cloud_colors = np.asarray(pcd.colors)
            point_cloud = np.concatenate((point_cloud, point_cloud_colors), axis=1)
            # print(f"Original Point Cloud shape: {point_cloud.shape}")
            if self.point_cloud_transform:
                point_cloud = self.point_cloud_transform(point_cloud)
            point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
            # print(f"Transformed Point Cloud shape: {point_cloud.shape}")
        else:
            point_cloud = torch.empty(0)
            # print(f"No Point Cloud data available for image id: {image_id}")

        label = torch.tensor(
            list(self.json_data[image_id].values()), dtype=torch.float32
        )
        # print(f"Label shape: {label.shape}")

        return rgbd_image, point_cloud, label
