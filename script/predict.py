import argparse
import os
import json
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from model import TomatoNet, TomatoViT
from transforms import transform


def load_model(model_path, model_type, num_traits=4):
    if model_type == "resnet50":
        model = TomatoNet(num_traits=num_traits)
    elif model_type == "vit":
        model = TomatoViT(num_traits=num_traits)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    )
    model.eval()
    return model


def preprocess_image(rgb_path, depth_path):
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


def predict(model, device, rgbd_image):
    with torch.no_grad():
        rgbd_image = rgbd_image.to(device)
        outputs = model(rgbd_image)
        outputs = outputs.cpu().numpy()[0]
    return outputs


def format_predictions(predictions):
    formatted_predictions = {
        "height": round(max(float(predictions[0]), 0), 1),
        "fw_plant": round(max(float(predictions[1]), 0), 2),
        "leaf_area": round(max(float(predictions[2]), 0), 2),
        "number_of_red_fruits": (
            round(max(float(predictions[3]), 0), 1) if predictions[3] >= 0 else 0.0
        ),
    }
    return formatted_predictions


def main():
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
        "--model_path", type=str, required=True, help="Path to the trained model."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet50",
        choices=["resnet50", "vit"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder to save the output JSON predictions.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, args.model_type).to(device)

    predictions = {}
    rgb_files = [f for f in os.listdir(args.rgb_folder) if f.endswith(".png")]
    total_images = len(rgb_files)

    with tqdm(total=total_images, desc="Processing images", unit="image") as pbar:
        for filename in rgb_files:
            image_id = filename.split(".")[0]
            rgb_path = os.path.join(args.rgb_folder, filename)
            depth_path = os.path.join(args.depth_folder, f"{image_id}_depth.png")

            if os.path.exists(rgb_path) and os.path.exists(depth_path):
                rgbd_image = preprocess_image(rgb_path, depth_path)
                preds = predict(model, device, rgbd_image)
                formatted_preds = format_predictions(preds)
                predictions[image_id] = formatted_preds

                # Update progress bar
                pbar.update(1)

    output_filename = f"{args.model_type}_output.json"
    output_path = os.path.join(args.output_folder, output_filename)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
