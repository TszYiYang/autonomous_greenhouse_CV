#/bin/bash

rgb_folder="--rgb_folder /Volume/greenhouse/rgb"

depth_folder="--depth_folder /Volume/greenhouse/depth"

output_folder="--output_folder /home/yangze2065/autonomous_greenhouse_CV/results"


python script/segment.py $rgb_folder $depth_folder $output_folder