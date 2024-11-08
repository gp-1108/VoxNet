#!/bin/bash

project_dir=/home/gp1108/Code/VoxNet
ext_path=/ext
dataset_archive="${ext_path}/ModelNet40Voxel.zip"
dataset_path="${ext_path}/ModelNet40Voxel"
model_output="${project_dir}/model.pth"
exec_script="${project_dir}/training/train.py"
sif_image_path=/ext/sif_image.sif
gdrive_sif_id=todo
gdrive_dataset_id=1Pjlcpcxsp1EtS60UG0DFgGqJ3xfycF0_

export PATH="$PATH:$(python3 -m site --user-base)/bin"

# Clean up the dataset directory before downloading
rm -rf $dataset_path $dataset_archive $model_output

# Download the dataset
gdown --id $gdrive_sif_id -O $sif_image_path
gdown --id $gdrive_dataset_id -O $dataset_archive

# Unzip the dataset
unzip -q $dataset_archive -d /ext

# Hyperparameters
batch_size=64
epochs=100
lr=0.001

# Run the script
cd $project_dir
singularity exec --nv --no-home -B $ext_path -B $project_dir $sif_image_path \
    python3 $exec_script \
    --dataset_path $dataset_path \
    --output_path $model_output \
    --batch_size $batch_size \
    --learning_rate $lr \
    --num_epochs $epochs

# Zip the dataset and save it
zip -r $save_path $dataset_output