"""
train.py

This module contains the training and validation logic for the TomatoNet model. 
It includes functions for computing metrics, training the model, evaluating the model, and plotting training results.

Functions:
- compute_metrics: Computes evaluation metrics for the model's predictions.
- train_model: Trains and validates the model, and saves checkpoints at regular intervals.
- evaluate_model: Evaluates the model on the test set.
- plot_metrics: Plots the training and validation losses and other metrics.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def compute_metrics(outputs, labels, threshold=0.5):
    """
    Computes evaluation metrics for the model's predictions.

    Args:
    - outputs (torch.Tensor): Model predictions.
    - labels (torch.Tensor): Ground truth labels.
    - threshold (float): Threshold for converting continuous outputs to binary classes.

    Returns:
    - tuple: Computed metrics (MSE, MAE, accuracy, F1 score, precision, recall).
    """
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    mse = mean_squared_error(labels, outputs)
    mae = mean_absolute_error(labels, outputs)

    # Convert continuous outputs to binary classes
    outputs_bin = (outputs > threshold).astype(int)
    labels_bin = (labels > threshold).astype(int)

    accuracy = accuracy_score(labels_bin, outputs_bin)
    f1 = f1_score(labels_bin, outputs_bin, average="macro")
    precision = precision_score(
        labels_bin, outputs_bin, average="macro", zero_division=0
    )
    recall = recall_score(labels_bin, outputs_bin, average="macro")

    return mse, mae, accuracy, f1, precision, recall


def plot_metrics(train_losses, val_losses, train_metrics, val_metrics, results_dir):
    """
    Plots the training and validation losses and other metrics.

    Args:
    - train_losses (list): List of training losses.
    - val_losses (list): List of validation losses.
    - train_metrics (list): List of training metrics.
    - val_metrics (list): List of validation metrics.
    - results_dir (str): Directory to save the plots.
    """
    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(results_dir, "plots", "losses.png"))
    plt.close()

    # Plot MSE
    train_mse = [m[0] for m in train_metrics]
    val_mse = [m[0] for m in val_metrics]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_mse, label="Training MSE")
    plt.plot(epochs, val_mse, label="Validation MSE")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.title("Training and Validation MSE")
    plt.savefig(os.path.join(results_dir, "plots", "mse.png"))
    plt.close()

    # Plot MAE
    train_mae = [m[1] for m in train_metrics]
    val_mae = [m[1] for m in val_metrics]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_mae, label="Training MAE")
    plt.plot(epochs, val_mae, label="Validation MAE")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.title("Training and Validation MAE")
    plt.savefig(os.path.join(results_dir, "plots", "mae.png"))
    plt.close()

    # Plot Accuracy
    train_accuracy = [m[2] for m in train_metrics]
    val_accuracy = [m[2] for m in val_metrics]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.savefig(os.path.join(results_dir, "plots", "accuracy.png"))
    plt.close()

    # Plot F1 Score
    train_f1 = [m[3] for m in train_metrics]
    val_f1 = [m[3] for m in val_metrics]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_f1, label="Training F1 Score")
    plt.plot(epochs, val_f1, label="Validation F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.title("Training and Validation F1 Score")
    plt.savefig(os.path.join(results_dir, "plots", "f1.png"))
    plt.close()

    # Plot Precision
    train_precision = [m[4] for m in train_metrics]
    val_precision = [m[4] for m in val_metrics]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_precision, label="Training Precision")
    plt.plot(epochs, val_precision, label="Validation Precision")
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Training and Validation Precision")
    plt.savefig(os.path.join(results_dir, "plots", "precision.png"))
    plt.close()

    # Plot Recall
    train_recall = [m[5] for m in train_metrics]
    val_recall = [m[5] for m in val_metrics]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_recall, label="Training Recall")
    plt.plot(epochs, val_recall, label="Validation Recall")
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend()
    plt.title("Training and Validation Recall")
    plt.savefig(os.path.join(results_dir, "plots", "recall.png"))
    plt.close()


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=1000,
    save_checkpoint_interval=100,
    device="cpu",
    results_dir="results",
    accumulation_steps=4,
):
    """
    Trains and validates the model, and saves checkpoints at regular intervals.

    Args:
    - model (nn.Module): The model to be trained.
    - train_loader (DataLoader): DataLoader for the training set.
    - val_loader (DataLoader): DataLoader for the validation set.
    - criterion (nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer.
    - scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    - num_epochs (int): Number of epochs for training.
    - save_checkpoint_interval (int): Interval for saving checkpoints.
    - device (str): Device to run the training on ('cpu' or 'cuda').
    - results_dir (str): Directory to save results, checkpoints, and models.
    - accumulation_steps (int): Number of steps to accumulate gradients before updating weights.

    Returns:
    - tuple: Lists of training losses, validation losses, training metrics, and validation metrics.
    """
    scaler = GradScaler()
    best_val_loss = float("inf")
    early_stopping_counter = 0
    early_stopping_patience = 10

    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    # Create directories for saving checkpoints, models, and plots
    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        (
            train_mse,
            train_mae,
            train_accuracy,
            train_f1,
            train_precision,
            train_recall,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        optimizer.zero_grad()

        for i, (rgbd_images, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        ):
            rgbd_images, labels = rgbd_images.to(device), labels.to(device)

            # Print the shapes of images and labels
            if epoch == 0 and i == 0:
                print(
                    f"Shapes - RGBD images: {rgbd_images.shape}, labels: {labels.shape}"
                )

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(rgbd_images)
                loss = criterion(outputs, labels)

            # Check for NaN values
            if torch.isnan(loss).any():
                print(f"NaN loss encountered at epoch {epoch+1}, batch {i+1}")
                continue

            # Backward pass with gradient accumulation
            scaler.scale(loss / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )  # Gradient clipping
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()

            # Compute metrics
            (
                batch_mse,
                batch_mae,
                batch_accuracy,
                batch_f1,
                batch_precision,
                batch_recall,
            ) = compute_metrics(outputs, labels)
            train_mse += batch_mse
            train_mae += batch_mae
            train_accuracy += batch_accuracy
            train_f1 += batch_f1
            train_precision += batch_precision
            train_recall += batch_recall

        epoch_loss = running_loss / len(train_loader)
        train_mse /= len(train_loader)
        train_mae /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_f1 /= len(train_loader)
        train_precision /= len(train_loader)
        train_recall /= len(train_loader)
        train_losses.append(epoch_loss)
        train_metrics.append(
            (
                train_mse,
                train_mae,
                train_accuracy,
                train_f1,
                train_precision,
                train_recall,
            )
        )
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}"
        )

        # Validation
        model.eval()
        val_loss = 0.0
        val_mse, val_mae, val_accuracy, val_f1, val_precision, val_recall = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        with torch.no_grad():
            for rgbd_images, labels in tqdm(
                val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"
            ):
                rgbd_images, labels = rgbd_images.to(device), labels.to(device)

                outputs = model(rgbd_images)
                loss = criterion(outputs, labels)

                # Check for NaN values
                if torch.isnan(loss).any():
                    print(f"NaN loss encountered at epoch {epoch+1}, validation batch")
                    continue

                val_loss += loss.item()

                # Compute metrics
                (
                    batch_mse,
                    batch_mae,
                    batch_accuracy,
                    batch_f1,
                    batch_precision,
                    batch_recall,
                ) = compute_metrics(outputs, labels)
                val_mse += batch_mse
                val_mae += batch_mae
                val_accuracy += batch_accuracy
                val_f1 += batch_f1
                val_precision += batch_precision
                val_recall += batch_recall

        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_mae /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_f1 /= len(val_loader)
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)
        val_losses.append(val_loss)
        val_metrics.append(
            (val_mse, val_mae, val_accuracy, val_f1, val_precision, val_recall)
        )
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}"
        )

        # Step the scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

        # Save checkpoint every save_checkpoint_interval epochs
        if (epoch + 1) % save_checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                results_dir, "checkpoints", f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save the final model and its weights
    final_model_path = os.path.join(results_dir, "models", "tomato_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    return train_losses, val_losses, train_metrics, val_metrics


def evaluate_model(model, test_loader, criterion, device="cpu"):
    """
    Evaluates the model on the test set and prints the metrics.

    Args:
    - model (nn.Module): The trained model to evaluate.
    - test_loader (DataLoader): DataLoader for the test set.
    - criterion (nn.Module): Loss function.
    - device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
    - dict: Dictionary containing the evaluation metrics.
    """
    model.eval()
    test_loss = 0.0
    test_mse, test_mae, test_accuracy, test_f1, test_precision, test_recall = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    with torch.no_grad():
        for rgbd_images, labels in tqdm(test_loader, desc="Evaluating"):
            rgbd_images, labels = rgbd_images.to(device), labels.to(device)

            outputs = model(rgbd_images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # Compute metrics
            (
                batch_mse,
                batch_mae,
                batch_accuracy,
                batch_f1,
                batch_precision,
                batch_recall,
            ) = compute_metrics(outputs, labels)
            test_mse += batch_mse
            test_mae += batch_mae
            test_accuracy += batch_accuracy
            test_f1 += batch_f1
            test_precision += batch_precision
            test_recall += batch_recall

    test_loss /= len(test_loader)
    test_mse /= len(test_loader)
    test_mae /= len(test_loader)
    test_accuracy /= len(test_loader)
    test_f1 /= len(test_loader)
    test_precision /= len(test_loader)
    test_recall /= len(test_loader)

    metrics = {
        "Test Loss": test_loss,
        "MSE": test_mse,
        "MAE": test_mae,
        "Accuracy": test_accuracy,
        "F1 Score": test_f1,
        "Precision": test_precision,
        "Recall": test_recall,
    }

    print(
        f"Test Loss: {test_loss:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}"
    )

    return metrics
