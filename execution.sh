#/bin/bash

rgb_folder="--rgb_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/data/train/rgb"

depth_folder="--depth_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/data/train/depth"

results_dir="--results_dir /home/yangze2065/Documents/autonomous_greenhouse_CV/results"

config_file="--config_file /home/yangze2065/Documents/autonomous_greenhouse_CV/results/oak-d-s2-poe_conf.json"   

json_path="--json_path /home/yangze2065/Documents/autonomous_greenhouse_CV/data/train/ground_truth_train.json"

pcd_folder="--pcd_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/results/point_cloud"

python script/execution.py $rgb_folder $depth_folder $output_folder $pcd_folder $json_path 