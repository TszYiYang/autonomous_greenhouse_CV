import argparse
from pathlib import Path
import open3d as o3d
import numpy as np
from pointcloud_tools import PointCloudCreator, create_pcd_from_array, write_pcd


def generate_point_cloud(rgb_file, depth_file, output_folder, config_file):
    point_cloud_creator = PointCloudCreator(conf_file=config_file)

    # Convert depth to point array
    points = point_cloud_creator.convert_depth_to_point_array(depth_file)

    # Create point cloud from array
    pcd = create_pcd_from_array(rgb_file, points)

    # Flip the z-axis by 180 degrees
    flip_transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd.transform(flip_transform)

    # Save the point cloud to the specified folder
    point_cloud_folder = output_folder / "point_cloud"
    point_cloud_folder.mkdir(parents=True, exist_ok=True)
    output_path = point_cloud_folder / f"{rgb_file.stem}_pcd.pcd"
    write_pcd(pcd, output_path)

    return np.asarray(pcd.points), np.asarray(pcd.colors)


def main():
    parser = argparse.ArgumentParser(
        description="Generate point clouds from RGB and depth images."
    )
    parser.add_argument(
        "--rgb_folder", required=True, help="Path to the folder containing RGB images."
    )
    parser.add_argument(
        "--depth_folder",
        required=True,
        help="Path to the folder containing depth images.",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        help="Path to the folder to save the point clouds.",
    )
    parser.add_argument(
        "--config_file",
        required=True,
        help="Path to the camera configuration JSON file.",
    )
    args = parser.parse_args()

    rgb_folder = Path(args.rgb_folder)
    depth_folder = Path(args.depth_folder)
    output_folder = Path(args.output_folder)

    for rgb_file in rgb_folder.glob("*.png"):
        depth_file = depth_folder / f"{rgb_file.stem}_depth.png"
        if depth_file.exists():
            generate_point_cloud(rgb_file, depth_file, output_folder, args.config_file)
        else:
            print(f"Depth file not found for {rgb_file}")


if __name__ == "__main__":
    main()
