#/bin/bash

rgb_folder="--rgb_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/data/train/rgb"

depth_folder="--depth_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/data/train/depth"

output_folder="--output_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/results"


python script/segment.py $rgb_folder $depth_folder $output_folder