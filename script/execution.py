import argparse
import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TomatoNet, TomatoViT, TomatoComboVitTomatoPCD, TomatoComboVitPointNet
from dataset import filter_json_data, TomatoDataset
from transforms import transform, point_cloud_transform
from train import train_model, evaluate_model, plot_metrics


def filter_json_data(json_path, rgb_folder, depth_folder):
    """
    Filter the JSON data to ensure corresponding RGB and depth images exist.

    Parameters:
        json_path (str): Path to the JSON file containing the data.
        rgb_folder (str): Path to the folder containing RGB images.
        depth_folder (str): Path to the folder containing depth images.

    Returns:
        filtered_data (dict): The filtered data dictionary.
        filtered_image_ids (list): List of filtered image IDs.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    filtered_data = {}
    filtered_image_ids = []
    for image_id, traits in data.items():
        rgb_path = os.path.join(rgb_folder, f"{image_id}.png")
        depth_path = os.path.join(depth_folder, f"{image_id}_depth.png")
        if os.path.exists(rgb_path) and os.path.exists(depth_path):
            filtered_data[image_id] = traits
            filtered_image_ids.append(image_id)

    return filtered_data, filtered_image_ids


def main():
    """
    Main function to train and evaluate the model (ResNet50, ViT, or Combo) on RGBD plant images.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate the model (ResNet50, ViT, or Combo) on RGBD plant images."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["resnet50", "vit", "combo_vit_tomatoPCD", "combo_vit_pointNet"],
        help="Type of model to load.",
    )
    parser.add_argument(
        "--rgb_folder",
        type=str,
        required=True,
        help="Path to the folder containing RGB images.",
    )
    parser.add_argument(
        "--depth_folder",
        type=str,
        required=True,
        help="Path to the folder containing depth images.",
    )
    parser.add_argument(
        "--pcd_folder",
        type=str,
        required=False,
        help="Path to the folder containing point cloud files.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the plant traits data.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of epochs for training the model.",
    )
    parser.add_argument(
        "--save_checkpoint_interval",
        type=int,
        default=100,
        help="Interval (in epochs) for saving model checkpoints.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save training results, checkpoints, and models.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps for training.",
    )
    parser.add_argument(
        "--pretrained_pointnet",
        type=str,
        help="Path to pretrained weights for the PointNet model (required for combo_vit_pointNet).",
    )

    args = parser.parse_args()

    # Filter JSON data to ensure corresponding RGB and depth images exist
    filtered_data, filtered_image_ids = filter_json_data(
        args.json_path, args.rgb_folder, args.depth_folder
    )

    if args.model_type in ["combo_vit_tomatoPCD", "combo_vit_pointNet"]:
        dataset = TomatoDataset(
            rgb_folder=args.rgb_folder,
            depth_folder=args.depth_folder,
            json_data=filtered_data,
            image_ids=filtered_image_ids,
            transform=transform,
            point_cloud_transform=point_cloud_transform,
            pcd_folder=args.pcd_folder,
        )
    else:
        dataset = TomatoDataset(
            rgb_folder=args.rgb_folder,
            depth_folder=args.depth_folder,
            json_data=filtered_data,
            image_ids=filtered_image_ids,
            transform=transform,
            point_cloud_transform=None,
            pcd_folder=None,
        )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    num_traits = 4  # Number of plant traits to predict
    if args.model_type == "resnet50":
        model = TomatoNet(num_traits=num_traits)
    elif args.model_type == "vit":
        model = TomatoViT(num_traits=num_traits)
    elif args.model_type == "combo_vit_tomatoPCD":
        model = TomatoComboVitTomatoPCD(num_traits=num_traits)
    elif args.model_type == "combo_vit_pointNet":
        model = TomatoComboVitPointNet(num_traits=num_traits)
        if args.pretrained_pointnet:
            model.pcd_model.load_pretrained_weights(args.pretrained_pointnet)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Define loss function and optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Train and evaluate the model
    train_losses, val_losses, train_metrics, val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        save_checkpoint_interval=args.save_checkpoint_interval,
        device=device,
        results_dir=args.results_dir,
        accumulation_steps=args.accumulation_steps,
        scaler=scaler,
        model_type=args.model_type,
    )

    # Plot metrics
    plot_metrics(train_losses, val_losses, train_metrics, val_metrics, args.results_dir)


if __name__ == "__main__":
    main()
