import argparse
import os
import json
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import open3d as o3d  # Import open3d for PCD file handling
from model import TomatoNet, TomatoViT, TomatoComboVitTomatoPCD, TomatoComboVitPointNet
from transforms import transform


def load_model(model_path, model_type, num_traits=4):
    """
    Load the specified model with the given model type and path.

    Parameters:
        model_path (str): Path to the trained model file.
        model_type (str): Type of the model to load.
        num_traits (int): Number of traits to predict.

    Returns:
        model (torch.nn.Module): The loaded model.
    """
    if model_type == "resnet50":
        model = TomatoNet(num_traits=num_traits)
    elif model_type == "vit":
        model = TomatoViT(num_traits=num_traits)
    elif model_type == "combo_vit_tomatoPCD":
        model = TomatoComboVitTomatoPCD(num_traits=num_traits)
    elif model_type == "combo_vit_pointNet":
        model = TomatoComboVitPointNet(num_traits=num_traits)
    else:
        raise ValueError("Unknown model type based on the provided model type")

    # Load the state_dict
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Adapt the state_dict to the new model structure
    model_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if v.size() == model_dict[k].size():
                new_state_dict[k] = v
            else:
                print(
                    f"Skipping resizing layer {k}: from {v.size()} to {model_dict[k].size()}"
                )
        else:
            print(f"Skipping unexpected layer {k}")

    # Initialize missing layers with default values
    for key in model_dict.keys():
        if key not in new_state_dict:
            print(f"Initializing missing key: {key}")
            new_state_dict[key] = model_dict[key]

    # Load the adapted state_dict
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def preprocess_image(rgb_path, depth_path):
    """
    Preprocess the RGB and depth images to create a 4-channel RGBD image.

    Parameters:
        rgb_path (str): Path to the RGB image file.
        depth_path (str): Path to the depth image file.

    Returns:
        rgbd_image (torch.Tensor): The preprocessed RGBD image tensor.
    """
    rgb_image = Image.open(rgb_path).convert("RGB")
    depth_image = Image.open(depth_path).convert("L")

    rgb_image = np.array(rgb_image)
    depth_image = np.array(depth_image)

    rgb_image = np.transpose(rgb_image, (2, 0, 1))
    depth_image = np.expand_dims(depth_image, axis=0)

    rgbd_image = np.concatenate((rgb_image, depth_image), axis=0)

    if transform:
        rgbd_image = transform(rgbd_image)

    rgbd_image = torch.tensor(rgbd_image, dtype=torch.float32).unsqueeze(
        0
    )  # Add batch dimension
    return rgbd_image


def preprocess_point_cloud(pcd_path):
    """
    Preprocess the point cloud data.

    Parameters:
        pcd_path (str): Path to the point cloud file.

    Returns:
        point_cloud (torch.Tensor): The preprocessed point cloud tensor.
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        point_cloud = np.hstack((points, colors))
    else:
        point_cloud = points

    point_cloud = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(
        0
    )  # Add batch dimension
    return point_cloud


def predict(model, device, rgbd_image, point_cloud=None):
    """
    Make predictions using the given model and input data.

    Parameters:
        model (torch.nn.Module): The trained model.
        device (torch.device): The device to run the predictions on.
        rgbd_image (torch.Tensor): The preprocessed RGBD image tensor.
        point_cloud (torch.Tensor, optional): The preprocessed point cloud tensor.

    Returns:
        outputs (numpy.ndarray): The prediction outputs.
    """
    with torch.no_grad():
        rgbd_image = rgbd_image.to(device)
        if point_cloud is not None:
            point_cloud = point_cloud.to(device)
            outputs = model(
                rgbd_image, point_cloud
            )  # Call with both rgbd_image and point_cloud
        else:
            outputs = model(
                rgbd_image
            )  # This would only work for models that don't need point cloud
        outputs = outputs.cpu().numpy()[0]
    return outputs


def format_predictions(predictions, num_traits=4):
    """
    Format the predictions into a dictionary with appropriate rounding.

    Parameters:
        predictions (numpy.ndarray): The raw prediction outputs.
        num_traits (int): Number of traits to format.

    Returns:
        formatted_predictions (dict): The formatted predictions.
    """
    if num_traits == 4:
        formatted_predictions = {
            "height": round(max(float(predictions[0]), 0), 1),
            "fw_plant": round(max(float(predictions[1]), 0), 2),
            "leaf_area": round(max(float(predictions[2]), 0), 2),
            "number_of_red_fruits": (
                round(max(float(predictions[3]), 0), 1) if predictions[3] >= 0 else 0.0
            ),
        }
    else:
        formatted_predictions = {
            f"trait_{i}": round(max(float(pred), 0), 2)
            for i, pred in enumerate(predictions)
        }
    return formatted_predictions


def main():
    """
    Main function to load the model, preprocess the input data, make predictions,
    and save the results to a JSON file.
    """
    parser = argparse.ArgumentParser(
        description="Predict plant traits using trained model."
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
        "--model_path", type=str, required=True, help="Path to the trained model."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["resnet50", "vit", "combo_vit_tomatoPCD", "combo_vit_pointNet"],
        help="Type of model to load.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder to save the output JSON predictions.",
    )
    args = parser.parse_args()

    # Ensure PCD folder is provided for combo models
    if (
        args.model_type in ["combo_vit_tomatoPCD", "combo_vit_pointNet"]
        and not args.pcd_folder
    ):
        raise ValueError("PCD folder is required for combo models.")

    num_traits = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, args.model_type, num_traits=num_traits).to(
        device
    )

    predictions = {}
    rgb_files = [f for f in os.listdir(args.rgb_folder) if f.endswith(".png")]
    total_images = len(rgb_files)

    with tqdm(total=total_images, desc="Processing images", unit="image") as pbar:
        for filename in rgb_files:
            image_id = filename.split(".")[0]
            rgb_path = os.path.join(args.rgb_folder, filename)
            depth_path = os.path.join(args.depth_folder, f"{image_id}_depth.png")
            pcd_path = (
                os.path.join(args.pcd_folder, f"{image_id}_pcd.pcd")
                if args.pcd_folder
                else None
            )

            if os.path.exists(rgb_path) and os.path.exists(depth_path):
                rgbd_image = preprocess_image(rgb_path, depth_path)
                point_cloud = (
                    preprocess_point_cloud(pcd_path)
                    if pcd_path and os.path.exists(pcd_path)
                    else None
                )
                preds = predict(model, device, rgbd_image, point_cloud)
                formatted_preds = format_predictions(preds, num_traits=num_traits)
                predictions[image_id] = formatted_preds
                pbar.update(1)

    output_filename = f"output_{args.model_type}.json"
    output_path = os.path.join(args.output_folder, output_filename)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
