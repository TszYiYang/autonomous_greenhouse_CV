"""
execution.py

This script performs the training and evaluation of the TomatoNet model. 
It supports customization of paths for RGB images, depth images, and ground truth JSON file.

Usage:
    python execution.py --rgb_folder <path_to_rgb_images> --depth_folder <path_to_depth_images> --json_path <path_to_ground_truth_json> --num_epochs <num_epochs> --save_checkpoint_interval <interval>

"""

import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import filter_json_data, TomatoDataset
from model import TomatoNet, TomatoViT, TomatoCombo
from train import train_model, evaluate_model, plot_metrics
from transforms import transform


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the model (ResNet50 or ViT) on RGBD plant images."
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
        required=True,
        help="Path to the folder containing point cloud files.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the ground truth JSON file.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1000, help="Number of epochs for training."
    )
    parser.add_argument(
        "--save_checkpoint_interval",
        type=int,
        default=100,
        help="Interval for saving checkpoints.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results, checkpoints, and models.",
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
        help="Number of steps to accumulate gradients before updating weights.",
    )
    args = parser.parse_args()

    # Filter the JSON data and get the filtered image IDs
    filtered_data, filtered_image_ids = filter_json_data(
        args.json_path, args.rgb_folder, args.depth_folder
    )

    # Create datasets and dataloaders
    train_size = int(0.8 * len(filtered_image_ids))
    val_size = int(0.1 * len(filtered_image_ids))
    test_size = len(filtered_image_ids) - train_size - val_size
    dataset = TomatoDataset(
        args.rgb_folder,
        args.depth_folder,
        args.pcd_folder,
        filtered_data,
        filtered_image_ids,
        transform=transform,
    )
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model, loss function, optimizer, and learning rate scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model selection for TomatoCombo
    model = TomatoCombo(num_traits=4).to(device)

    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )

    # Train the model
    train_losses, val_losses, train_metrics, val_metrics = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=args.num_epochs,
        save_checkpoint_interval=args.save_checkpoint_interval,
        device=device,
        results_dir=args.results_dir,
        accumulation_steps=args.accumulation_steps,
    )

    # Evaluate the model
    test_metrics = evaluate_model(model, test_loader, criterion, device=device)

    # Plot the metrics
    plot_metrics(train_losses, val_losses, train_metrics, val_metrics, args.results_dir)


if __name__ == "__main__":
    main()
