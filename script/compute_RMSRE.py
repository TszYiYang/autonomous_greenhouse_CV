import json
import numpy as np
import argparse


def calculate_error(pred: dict, truth: dict) -> float:
    """
    Calculate the average Root Mean Squared Relative Error (RMSRE)
    between predicted and ground truth values for each trait.

    Args:
    pred (dict): Dictionary containing predicted trait values.
    truth (dict): Dictionary containing ground truth trait values.

    Returns:
    float: The average RMSRE across all traits.
    """
    error = 0
    for trait in ["height", "fw_plant", "leaf_area", "number_of_red_fruits"]:
        diff = [
            ((pred[i][trait] - truth[i][trait]) / (truth[i][trait] + 1)) ** 2
            for i in truth
        ]
        error += np.sqrt(np.nanmean(diff))
    return error / 4


def load_json(file_path: str) -> dict:
    """
    Load a JSON file and return its contents as a dictionary.

    Args:
    file_path (str): Path to the JSON file.

    Returns:
    dict: The contents of the JSON file.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def main(groundtruth_path: str, predict_path: str):
    """
    Main function to load JSON files, calculate the error, and print the result.

    Args:
    groundtruth_path (str): Path to the ground truth JSON file.
    predict_path (str): Path to the predicted JSON file.
    """
    groundtruth_data = load_json(groundtruth_path)
    predict_data = load_json(predict_path)

    error = calculate_error(predict_data, groundtruth_data)
    print(f"Average RMSRE: {error:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute average RMSRE between ground truth and predicted JSON files."
    )
    parser.add_argument(
        "groundtruth_path", type=str, help="Path to the ground truth JSON file."
    )
    parser.add_argument(
        "predict_path", type=str, help="Path to the predicted JSON file."
    )

    args = parser.parse_args()

    main(args.groundtruth_path, args.predict_path)
