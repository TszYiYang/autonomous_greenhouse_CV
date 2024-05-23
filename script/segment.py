import argparse
import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


def load_images_from_folder(folder, suffix):
    return [
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if filename.endswith(suffix)
    ]


def resize_image(image, size):
    return image.resize(size, Image.LANCZOS)


def setup_predictor(model_name, score_thresh):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # COCO dataset has 80 classes
    predictor = DefaultPredictor(cfg)
    return predictor


def is_centered(box, image_width, image_height, tolerance=0.35):
    x1, y1, x2, y2 = box
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2

    center_region_x_min = image_width * (0.5 - tolerance)
    center_region_x_max = image_width * (0.5 + tolerance)
    center_region_y_min = image_height * (0.5 - tolerance)
    center_region_y_max = image_height * (0.5 + tolerance)

    return (center_region_x_min <= box_center_x <= center_region_x_max) and (
        center_region_y_min <= box_center_y <= center_region_y_max
    )


def segment_instance(
    rgb_image, predictor, initial_score_thresh, min_score_thresh, image_id
):
    image_array = np.array(rgb_image)
    image_height, image_width = image_array.shape[:2]
    score_thresh = initial_score_thresh

    while score_thresh >= min_score_thresh:
        predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        outputs = predictor(image_array)
        masks = outputs["instances"].pred_masks.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

        potted_plant_mask = None
        centered_boxes = []

        for mask, class_id, box in zip(masks, classes, boxes):
            if class_id == 58:  # Class ID for "potted plant"
                potted_plant_mask = mask
                if is_centered(box, image_width, image_height):
                    return potted_plant_mask.astype(np.uint8)
            elif is_centered(box, image_width, image_height):
                centered_boxes.append(box)

        score_thresh -= 0.1

    if centered_boxes:
        print(
            f"No potted plant detected in the center for image {image_id} even at minimum threshold {min_score_thresh:.2f}. Using centered instance bounding box for masking."
        )
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        for box in centered_boxes:
            x1, y1, x2, y2 = map(int, box)
            combined_mask[y1:y2, x1:x2] = 1
        return combined_mask.astype(np.uint8)

    print(
        f"No instances detected in the center for image {image_id}. Using rectangular mask from center."
    )
    center_x, center_y = image_width // 2, image_height // 2
    rect_size_x = int(image_width * 0.6)
    rect_size_y = int(image_height * 0.6)
    x1, y1 = center_x - rect_size_x // 2, center_y - rect_size_y // 2
    x2, y2 = center_x + rect_size_x // 2, center_y + rect_size_y // 2
    rect_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    rect_mask[y1:y2, x1:x2] = 1
    return rect_mask


def apply_mask(image, mask):
    masked_image = np.zeros_like(image)
    if image.ndim == 3:  # RGB image
        for i in range(3):  # Apply the mask to each channel
            masked_image[:, :, i] = np.where(mask == 1, image[:, :, i], 0)
    else:  # Depth image
        masked_image = np.where(mask == 1, image, 0)
    return masked_image


def save_image(image, output_folder, image_id, suffix):
    """
    Saves the image to the specified output folder with a given naming convention.

    Args:
        image (np.ndarray): The image to save.
        output_folder (str): The folder to save the image in.
        image_id (str): The ID for naming the image file.
        suffix (str): The suffix for naming the image file.
    """
    if suffix:
        output_path = os.path.join(output_folder, f"{image_id}{suffix}.png")
    else:
        output_path = os.path.join(output_folder, f"{image_id}.png")
    Image.fromarray(image).save(output_path)


def main(
    rgb_folder,
    depth_folder,
    output_folder,
    model_name,
    initial_score_thresh,
    min_score_thresh,
):
    """
    Main function to process images, apply instance segmentation, and save masked RGB and depth images.

    Args:
        rgb_folder (str): Path to the folder containing RGB images.
        depth_folder (str): Path to the folder containing depth images.
        output_folder (str): Path to the folder to save masked images.
        model_name (str): Name of the pre-trained Detectron2 model to use.
        initial_score_thresh (float): Initial score threshold for predictions.
        min_score_thresh (float): Minimum score threshold for predictions.
    """
    rgb_images = load_images_from_folder(rgb_folder, ".png")
    depth_images = load_images_from_folder(depth_folder, "_depth.png")

    masked_rgb_output_folder = os.path.join(output_folder, "masked_rgb")
    masked_depth_output_folder = os.path.join(output_folder, "masked_depth")
    mask_output_folder = os.path.join(output_folder, "mask")

    os.makedirs(masked_rgb_output_folder, exist_ok=True)
    os.makedirs(masked_depth_output_folder, exist_ok=True)
    os.makedirs(mask_output_folder, exist_ok=True)

    predictor = setup_predictor(model_name, initial_score_thresh)

    for rgb_image_path in tqdm(rgb_images):
        image_id = os.path.splitext(os.path.basename(rgb_image_path))[0]
        depth_image_path = os.path.join(depth_folder, image_id + "_depth.png")

        if not os.path.exists(depth_image_path):
            print(f"Depth image for {rgb_image_path} not found. Skipping...")
            continue

        rgb_image = Image.open(rgb_image_path).convert("RGB")
        depth_image = Image.open(depth_image_path)

        # Resize depth image to match the RGB image size
        depth_image = resize_image(depth_image, rgb_image.size)

        rgb_array = np.array(rgb_image)
        depth_array = np.array(depth_image)

        mask = segment_instance(
            rgb_image, predictor, initial_score_thresh, min_score_thresh, image_id
        )

        masked_rgb_image = apply_mask(rgb_array, mask)
        masked_depth_image = apply_mask(depth_array, mask)

        # Save the masked RGB and depth images along with the mask
        save_image(masked_rgb_image, masked_rgb_output_folder, image_id, "")
        save_image(masked_depth_image, masked_depth_output_folder, image_id, "_depth")
        save_image(mask * 255, mask_output_folder, image_id, "_mask")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process RGB and depth images with Mask R-CNN using Detectron2."
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
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder to save masked images.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        help="Pre-trained Detectron2 model to use.",
    )
    parser.add_argument(
        "--initial_score_thresh",
        type=float,
        default=0.5,
        help="Initial score threshold for predictions.",
    )
    parser.add_argument(
        "--min_score_thresh",
        type=float,
        default=0.2,
        help="Minimum score threshold for predictions.",
    )

    args = parser.parse_args()
    main(
        args.rgb_folder,
        args.depth_folder,
        args.output_folder,
        args.model_name,
        args.initial_score_thresh,
        args.min_score_thresh,
    )
