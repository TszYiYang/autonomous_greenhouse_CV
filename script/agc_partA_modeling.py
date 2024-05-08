import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from PIL import Image
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import math


# Dataset class for handling JSON input data
class PlantDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file) as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.keys = list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.keys[idx]
        img_path = os.path.join(self.img_dir, img_id + ".png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        traits = {
            trait: self.data[img_id][trait]
            for trait in ["height", "fw_plant", "leaf_area", "number_of_red_fruits"]
        }
        return image, traits


# Define the model architecture
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        base_model = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return x


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 4)  # Predicting 4 traits
        )

    def forward(self, x):
        return self.regressor(x)


class PlantModel(nn.Module):
    def __init__(self):
        super(PlantModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.regressor = Regressor()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regressor(x)
        return x


# Define RMSRE calculation function
def calculate_rmsre(pred: dict, truth: dict) -> float:
    rmsre_total = 0
    trait_count = 0
    for trait in ["height", "fw_plant", "leaf_area", "number_of_red_fruits"]:
        errors = []
        for key in truth:
            if trait in truth[key] and trait in pred[key]:
                true_value = truth[key][trait]
                pred_value = pred[key][trait]
                if true_value is not None and pred_value is not None:
                    relative_error = (pred_value - true_value) / (true_value + 1)
                    squared_error = relative_error**2
                    errors.append(squared_error)
        if errors:
            mean_squared_error = np.nanmean(errors)
            rmsre = np.sqrt(mean_squared_error)
            rmsre_total += rmsre
            trait_count += 1
    average_rmsre = rmsre_total / trait_count if trait_count else float("inf")
    return average_rmsre


# Training function
def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    num_epochs=25,
    patience=5,
):
    train_losses = []
    val_rmsres = []
    best_rmsre = float("inf")  # Initialize best_rmsre before the loop
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, traits in train_loader:
            images = images.to(device)
            targets = (
                torch.tensor(
                    [
                        list(traits[trait])
                        for trait in [
                            "height",
                            "fw_plant",
                            "leaf_area",
                            "number_of_red_fruits",
                        ]
                    ]
                )
                .float()
                .to(device)
            )
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets.t())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate after each epoch
        val_rmsre = validate_model(val_loader, model, device)
        val_rmsres.append(val_rmsre)
        avg_rmsre = sum(val_rmsre.values()) / len(
            val_rmsre
        )  # Average RMSRE over traits for simplicity

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Avg Val RMSRE: {avg_rmsre:.4f}"
        )

        # Check for early stopping and save best model
        if avg_rmsre < best_rmsre:
            best_rmsre = avg_rmsre
            epochs_no_improve = 0
            best_model = model.state_dict()  # Save the current model state
            torch.save(
                best_model, f"best_model_epoch_{epoch+1}.pt"
            )  # Save the best model
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early stopping triggered.")
            early_stop = True
            break

    if not early_stop:
        print("Reached maximum number of epochs without early stopping.")

    return train_losses, val_rmsres


# Validation function with RMSRE calculation
def validate_model(data_loader, model, device):
    model.eval()
    total_squared_relative_errors = {
        trait: 0.0
        for trait in ["height", "fw_plant", "leaf_area", "number_of_red_fruits"]
    }
    total_count = 0

    with torch.no_grad():
        for images, traits in data_loader:
            images = images.to(device)
            outputs = model(images)
            outputs = (
                outputs.cpu().numpy()
            )  # Convert outputs to numpy array for easier manipulation

            # Calculate relative error for each trait
            for i, output in enumerate(outputs):
                actual_traits = [
                    traits[trait][i]
                    for trait in [
                        "height",
                        "fw_plant",
                        "leaf_area",
                        "number_of_red_fruits",
                    ]
                ]
                for j, trait in enumerate(
                    ["height", "fw_plant", "leaf_area", "number_of_red_fruits"]
                ):
                    if actual_traits[j] != 0:  # To avoid division by zero
                        rel_error = (output[j] - actual_traits[j]) / actual_traits[j]
                        total_squared_relative_errors[trait] += rel_error**2

            total_count += len(outputs)

    rmsre = {
        trait: math.sqrt(total_squared_relative_errors[trait] / total_count)
        for trait in total_squared_relative_errors
    }
    return rmsre


def plot_training(
    train_losses,
    val_rmsres,
    total_epochs=25,
    stopped_epoch=None,
    base_dir="/home/yangze2065/Documents/autonomous_greenhouse_challenge_2024/script/results/",
    base_filename="RMSE",
):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 7))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "bo-", label="Training Loss")
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot validation RMSRE for each trait
    plt.subplot(1, 2, 2)
    for trait in val_rmsres[0].keys():
        plt.plot(
            epochs, [rmsre[trait] for rmsre in val_rmsres], "o-", label=f"{trait} RMSRE"
        )
    plt.title("Validation RMSRE per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("RMSRE")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Ensure the directory exists, create if it does not
    import os

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Determine the filename based on whether early stopping occurred
    if stopped_epoch:
        filename = (
            f"{base_dir}{base_filename}_{total_epochs}_earlystop_at_{stopped_epoch}.png"
        )
    else:
        filename = f"{base_dir}{base_filename}_{total_epochs}_completed.png"

    # Save the figure
    plt.savefig(filename)
    print(f"Plot saved as {filename}")


if __name__ == "__main__":
    # Setup DataLoader for training and validation data
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = PlantDataset(
        "/home/yangze2065/Documents/autonomous_greenhouse_challenge_2024/cleaned_ground_truth_data.json",
        "/home/yangze2065/Documents/autonomous_greenhouse_challenge_2024/dataset/4th_dwarf_tomato/image/train/rgb",
        transform=transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    val_dataset = PlantDataset(
        "/home/yangze2065/Documents/autonomous_greenhouse_challenge_2024/dataset/4th_dwarf_tomato/image/validation/ground_truth_validation1.json",
        "/home/yangze2065/Documents/autonomous_greenhouse_challenge_2024/dataset/4th_dwarf_tomato/image/validation/rgb",
        transform=transform,
    )
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4)

    # Initialize model, loss, optimizer, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Run training and validation
    train_losses, val_rmsres = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        num_epochs=25,
    )

    # Plot the training and validation losses
    plot_training(train_losses, val_rmsres)
