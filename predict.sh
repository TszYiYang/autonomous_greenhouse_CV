#/bin/bash

rgb_folder="--rgb_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/data/validation/rgb"

depth_folder="--depth_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/data/validation/depth"

pcd_folder="--pcd_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/data/validation/point_cloud"

output_folder="--output_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/results/predictions"

model_path="--model_path /home/yangze2065/Documents/autonomous_greenhouse_CV/saved_model_results/model_trained_weights/tomato_model_combo_vit_pointNet.pth"

model_type="--model_type combo_vit_pointNet" # "resnet50", "vit", "combo_vit_tomatoPCD", "combo_vit_pointNet" #

python script/predict.py $rgb_folder $depth_folder $pcd_folder $output_folder $model_path $model_type

