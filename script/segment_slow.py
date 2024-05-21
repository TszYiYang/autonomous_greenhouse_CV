import argparse
import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


# Detailed descriptions included in the code


def load_images_from_folder(folder, suffix):
    """
    Loads images from the specified folder with a given suffix.

    Args:
        folder (str): Path to the folder containing images.
        suffix (str): Suffix of the image files to load.

    Returns:
        list: List of image file paths.
    """
    return [
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if filename.endswith(suffix)
    ]


def resize_image(image, size):
    """
    Resizes the image to the specified size.

    Args:
        image (PIL.Image): The image to resize.
        size (tuple): The target size (width, height).

    Returns:
        PIL.Image: The resized image.
    """
    return image.resize(size, Image.LANCZOS)


def setup_predictor(model_name, score_thresh):
    """
    Sets up the Detectron2 predictor with the specified model and score threshold.

    Args:
        model_name (str): Name of the pre-trained Detectron2 model to use.
        score_thresh (float): Score threshold for predictions.

    Returns:
        DefaultPredictor: The Detectron2 predictor.
    """
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # COCO dataset has 80 classes
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def is_centered(box, image_width, image_height, tolerance=0.35):
    """
    Checks if the bounding box is relatively centered in the image.

    Args:
        box (tuple): Bounding box coordinates (x1, y1, x2, y2).
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        tolerance (float): Tolerance for defining the central region. Default is 0.35.

    Returns:
        bool: True if the box is centered, False otherwise.
    """
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
    """
    Performs instance segmentation on the RGB image using Detectron2 with dynamic threshold adjustment.

    Args:
        rgb_image (PIL.Image): The RGB image.
        predictor (DefaultPredictor): The Detectron2 predictor.
        initial_score_thresh (float): Initial score threshold for predictions.
        min_score_thresh (float): Minimum score threshold for predictions.
        image_id (str): The ID of the current image.

    Returns:
        tuple: The mask for the "potted plant" class or bounding box mask if potted plant is not found, and the outputs from the predictor.
    """
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
                    return potted_plant_mask.astype(np.uint8), outputs
            elif is_centered(box, image_width, image_height):
                centered_boxes.append(box)

        print(
            f"No potted plant detected in the center at threshold {score_thresh:.2f}. Trying again with {score_thresh - 0.10:.2f}..."
        )
        score_thresh -= 0.10

    if centered_boxes:
        print(
            f"No potted plant detected in the center for image {image_id} even at minimum threshold {min_score_thresh:.2f}. Using centered instance bounding box for masking."
        )
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        for box in centered_boxes:
            x1, y1, x2, y2 = map(int, box)
            combined_mask[y1:y2, x1:x2] = 1
        return combined_mask.astype(np.uint8), outputs

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
    return rect_mask, outputs


def apply_mask(image, mask):
    """
    Applies the mask to the image, retaining pixel values in the masked area and setting the unmasked area to 0.

    Args:
        image (np.ndarray): The image to be masked.
        mask (np.ndarray): The mask to apply.

    Returns:
        np.ndarray: The masked image.
    """
    if image.ndim == 3:  # RGB image
        masked_image = image * mask[:, :, None]
    else:  # Depth image
        masked_image = image * mask
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
    output_path = os.path.join(output_folder, f"{image_id}_{suffix}.png")
    Image.fromarray(image.astype(np.uint8)).save(output_path)


def save_segmented_image(rgb_image, outputs, metadata, output_folder, image_id):
    """
    Saves the RGB image with detected segmentation, class names, and bounding boxes.

    Args:
        rgb_image (PIL.Image): The original RGB image.
        outputs (dict): The outputs from the predictor.
        metadata: Metadata for visualization.
        output_folder (str): The folder to save the segmented image in.
        image_id (str): The ID for naming the image file.
    """
    v = Visualizer(
        np.array(rgb_image)[:, :, ::-1],
        metadata,
        scale=1.2,
        instance_mode=ColorMode.IMAGE_BW,
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    segmented_image = v.get_image()[:, :, ::-1]
    output_path = os.path.join(output_folder, f"{image_id}_segmented.png")
    Image.fromarray(segmented_image).save(output_path)


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

    rgb_output_folder = os.path.join(output_folder, "masked_rgb")
    depth_output_folder = os.path.join(output_folder, "masked_depth")
    segmented_output_folder = os.path.join(output_folder, "segmented_image")

    os.makedirs(rgb_output_folder, exist_ok=True)
    os.makedirs(depth_output_folder, exist_ok=True)
    os.makedirs(segmented_output_folder, exist_ok=True)

    predictor, cfg = setup_predictor(model_name, initial_score_thresh)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    for rgb_image_path in tqdm(rgb_images):
        image_id = os.path.splitext(os.path.basename(rgb_image_path))[0]
        depth_image_path = os.path.join(depth_folder, image_id + "_depth.png")

        if not os.path.exists(depth_image_path):
            print(f"Depth image for {rgb_image_path} not found. Skipping...")
            continue

        rgb_image = Image.open(rgb_image_path).convert("RGB")
        depth_image = Image.open(depth_image_path).convert("L")

        # Resize depth image to match the RGB image size
        depth_image = resize_image(depth_image, rgb_image.size)

        rgb_array = np.array(rgb_image)
        depth_array = np.array(depth_image)

        mask, outputs = segment_instance(
            rgb_image, predictor, initial_score_thresh, min_score_thresh, image_id
        )

        masked_rgb_image = apply_mask(rgb_array, mask)
        masked_depth_image = apply_mask(depth_array, mask)

        save_image(masked_rgb_image, rgb_output_folder, image_id, "masked_rgb")
        save_image(masked_depth_image, depth_output_folder, image_id, "masked_depth")
        save_segmented_image(
            rgb_image, outputs, metadata, segmented_output_folder, image_id
        )


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
        default=0.50,
        help="Initial score threshold for predictions.",
    )
    parser.add_argument(
        "--min_score_thresh",
        type=float,
        default=0.30,
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
