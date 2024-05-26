#/bin/bash

rgb_folder="--rgb_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/data/validation/rgb"

depth_folder="--depth_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/data/validation/depth"

output_folder="--output_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/data/validation"

config_file="--config_file /home/yangze2065/Documents/autonomous_greenhouse_CV/data/validation/oak-d-s2-poe_conf.json"   


python script/pcd_gen.py $rgb_folder $depth_folder $output_folder $config_file