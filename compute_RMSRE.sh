#/bin/bash

groundtruth_validation=/home/yangze2065/Documents/autonomous_greenhouse_CV/data/validation/ground_truth_validation1.json 
test_validation=/home/yangze2065/Documents/autonomous_greenhouse_CV/data/validation/resnet50_masked_output.json

python script/compute_RMSRE.py $groundtruth_validation $test_validation