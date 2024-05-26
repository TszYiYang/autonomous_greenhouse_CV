#/bin/bash

rgb_folder="--rgb_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/data/validation/rgb"

depth_folder="--depth_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/data/validation/depth"

pcd_folder="--pcd_folder /home/yangze2065/Documents/autonomous_greenhouse_CV/data/validation/point_cloud"

json_path="--json_path /home/yangze2065/Documents/autonomous_greenhouse_CV/data/validation/ground_truth_validation1.json"

results_dir="--results_dir /home/yangze2065/Documents/autonomous_greenhouse_CV/results"

pretrained_pointnet="--pretrained_pointnet saved_model_results/model_trained_weights/pointNet_preTrained_2017paper.pth"

num_epochs="--num_epochs 1000" # default = 1000 epoch 

save_checkpoint_interval="--save_checkpoint_interval 50" # #default = 50 epoch

batch_size="--batch_size 2" # default = 16

accumulation_steps="--accumulation_steps 4" # default = 4

model_type="--model_type vit" # "resnet50", "vit", "combo_vit_tomatoPCD", "combo_vit_pointNet" 

python script/execution.py $rgb_folder $depth_folder $pcd_folder $json_path $results_dir $pretrained_pointnet $num_epochs $save_checkpoint_interval $batch_size $accumulation_steps $model_type

