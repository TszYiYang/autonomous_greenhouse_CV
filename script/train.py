import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def get_current_lr(optimizer):
    """
    Get the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer.

    Returns:
        float: The current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def check_for_nans(tensor, name="tensor"):
    """
    Check for NaN values in the tensor.

    Args:
        tensor (torch.Tensor): The tensor to check.
        name (str, optional): Name of the tensor for error message.

    Raises:
        ValueError: If NaN values are found in the tensor.
    """
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN values found in {name}")


def compute_metrics(outputs, labels):
    """
    Compute various metrics for the given outputs and labels.

    Args:
        outputs (torch.Tensor): The model outputs.
        labels (torch.Tensor): The ground truth labels.

    Returns:
        tuple: Computed MSE, RMSE, MAE, and R2 score.
    """
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    mse = mean_squared_error(labels, outputs)
    rmse = root_mean_squared_error(labels, outputs)
    mae = mean_absolute_error(labels, outputs)
    r2 = r2_score(labels, outputs)

    return mse, rmse, mae, r2


def plot_metrics(train_losses, val_losses, train_metrics, val_metrics, results_dir):
    """
    Plot and save various training and validation metrics.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        train_metrics (list): List of training metrics.
        val_metrics (list): List of validation metrics.
        results_dir (str): Directory to save the plots.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(results_dir, "plots", "losses.png"))
    plt.close()

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

    train_rmse = [m[1] for m in train_metrics]
    val_rmse = [m[1] for m in val_metrics]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_rmse, label="Training RMSE")
    plt.plot(epochs, val_rmse, label="Validation RMSE")
    plt.xlabel("Epochs")
    plt.ylabel("Root Mean Squared Error")
    plt.legend()
    plt.title("Training and Validation RMSE")
    plt.savefig(os.path.join(results_dir, "plots", "rmse.png"))
    plt.close()

    train_mae = [m[2] for m in train_metrics]
    val_mae = [m[2] for m in val_metrics]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_mae, label="Training MAE")
    plt.plot(epochs, val_mae, label="Validation MAE")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.title("Training and Validation MAE")
    plt.savefig(os.path.join(results_dir, "plots", "mae.png"))
    plt.close()

    train_r2 = [m[3] for m in train_metrics]
    val_r2 = [m[3] for m in val_metrics]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_r2, label="Training R2")
    plt.plot(epochs, val_r2, label="Validation R2")
    plt.xlabel("Epochs")
    plt.ylabel("R2 Score")
    plt.legend()
    plt.title("Training and Validation R2")
    plt.savefig(os.path.join(results_dir, "plots", "r2.png"))
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
    scaler=None,
    model_type="resnet50",
):
    """
    Train the model and evaluate on the validation set.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        num_epochs (int, optional): Number of epochs to train.
        save_checkpoint_interval (int, optional): Interval to save model checkpoints.
        device (torch.device, optional): Device to use for training.
        results_dir (str, optional): Directory to save results and checkpoints.
        accumulation_steps (int, optional): Number of gradient accumulation steps.
        scaler (torch.cuda.amp.GradScaler, optional): GradScaler for mixed precision training.
        model_type (str, optional): Type of model being trained.

    Returns:
        tuple: Lists of training and validation losses, and training and validation metrics.
    """
    if scaler is None:
        scaler = GradScaler()
    best_val_loss = float("inf")
    early_stopping_counter = 0
    early_stopping_patience = 20

    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    # Initialize TensorBoard writer
    log_dir = "runs/experiment_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir)

    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_mse, train_rmse, train_mae, train_r2 = 0.0, 0.0, 0.0, 0.0
        optimizer.zero_grad()

        for i, (rgbd_image, point_cloud, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        ):
            rgbd_image, labels = rgbd_image.to(device), labels.to(device)
            if model_type in ["combo_vit_tomatoPCD", "combo_vit_pointNet"]:
                point_cloud = point_cloud.to(device)
                if point_cloud.numel() == 0:
                    raise ValueError(
                        f"Point cloud data missing for combo model at index {i}"
                    )
            else:
                point_cloud = None

            if epoch == 0 and i == 0:
                print(
                    f"Shapes - RGBD images: {rgbd_image.shape}, Point Cloud: {point_cloud.shape if point_cloud is not None else 'None'}, Labels: {labels.shape}"
                )

            with torch.cuda.amp.autocast():
                if point_cloud is not None:
                    outputs = model(rgbd_image, point_cloud)
                else:
                    outputs = model(rgbd_image)
                loss = criterion(outputs, labels)

            if torch.isnan(loss).any():
                print(f"NaN loss encountered at epoch {epoch+1}, batch {i+1}")
                continue

            scaler.scale(loss / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()

            batch_mse, batch_rmse, batch_mae, batch_r2 = compute_metrics(
                outputs, labels
            )
            train_mse += batch_mse
            train_rmse += batch_rmse
            train_mae += batch_mae
            train_r2 += batch_r2

        epoch_loss = running_loss / len(train_loader)
        train_mse /= len(train_loader)
        train_rmse /= len(train_loader)
        train_mae /= len(train_loader)
        train_r2 /= len(train_loader)
        train_losses.append(epoch_loss)
        train_metrics.append((train_mse, train_rmse, train_mae, train_r2))
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}"
        )

        # Log training metrics to TensorBoard
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("MSE/Train", train_mse, epoch)
        writer.add_scalar("RMSE/Train", train_rmse, epoch)
        writer.add_scalar("MAE/Train", train_mae, epoch)
        writer.add_scalar("R2/Train", train_r2, epoch)

        # Log learning rate to TensorBoard
        writer.add_scalar("Learning Rate", get_current_lr(optimizer), epoch)

        model.eval()
        val_loss = 0.0
        val_mse, val_rmse, val_mae, val_r2 = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for rgbd_image, point_cloud, labels in tqdm(
                val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"
            ):
                rgbd_image, labels = rgbd_image.to(device), labels.to(device)
                if model_type in ["combo_vit_tomatoPCD", "combo_vit_pointNet"]:
                    point_cloud = point_cloud.to(device)
                    if point_cloud.numel() == 0:
                        raise ValueError(
                            f"Point cloud data missing for combo model at index {i}"
                        )
                else:
                    point_cloud = None

                if point_cloud is not None:
                    outputs = model(rgbd_image, point_cloud)
                else:
                    outputs = model(rgbd_image)
                loss = criterion(outputs, labels)

                if torch.isnan(loss).any():
                    print(f"NaN loss encountered at epoch {epoch+1}, validation batch")
                    continue

                val_loss += loss.item()

                batch_mse, batch_rmse, batch_mae, batch_r2 = compute_metrics(
                    outputs, labels
                )
                val_mse += batch_mse
                val_rmse += batch_rmse
                val_mae += batch_mae
                val_r2 += batch_r2

        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_rmse /= len(val_loader)
        val_mae /= len(val_loader)
        val_r2 /= len(val_loader)
        val_losses.append(val_loss)
        val_metrics.append((val_mse, val_rmse, val_mae, val_r2))
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}"
        )

        # Log validation metrics to TensorBoard
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("MSE/Validation", val_mse, epoch)
        writer.add_scalar("RMSE/Validation", val_rmse, epoch)
        writer.add_scalar("MAE/Validation", val_mae, epoch)
        writer.add_scalar("R2/Validation", val_r2, epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

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

    final_model_path = os.path.join(results_dir, "models", "tomato_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    writer.close()

    return train_losses, val_losses, train_metrics, val_metrics


def evaluate_model(model, test_loader, criterion, device="cpu"):
    """
    Evaluate the model on the test set.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for test data.
        criterion (nn.Module): Loss function.
        device (torch.device, optional): Device to use for evaluation.

    Returns:
        dict: Computed test metrics.
    """
    model.eval()
    test_loss = 0.0
    test_mse, test_rmse, test_mae, test_r2 = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for rgbd_image, point_cloud, labels in tqdm(test_loader, desc="Evaluating"):
            rgbd_image, labels = rgbd_image.to(device), labels.to(device)
            if point_cloud is not None:
                point_cloud = point_cloud.to(device)

            if point_cloud is not None:
                outputs = model(rgbd_image, point_cloud)
            else:
                outputs = model(rgbd_image)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            batch_mse, batch_rmse, batch_mae, batch_r2 = compute_metrics(
                outputs, labels
            )
            test_mse += batch_mse
            test_rmse += batch_rmse
            test_mae += batch_mae
            test_r2 += batch_r2

    test_loss /= len(test_loader)
    test_mse /= len(test_loader)
    test_rmse /= len(test_loader)
    test_mae /= len(test_loader)
    test_r2 /= len(test_loader)

    metrics = {
        "Test Loss": test_loss,
        "MSE": test_mse,
        "RMSE": test_rmse,
        "MAE": test_mae,
        "R2": test_r2,
    }

    print(
        f"Test Loss: {test_loss:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}"
    )

    return metrics
