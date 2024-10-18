#!/bin/bash

project_dir=/home/gp1108/Code/VoxNet
dataset_archive=/ext/ModelNet40.zip
dataset_path=/ext/ModelNet40
dataset_output=/ext/ModelNet40Voxel
sif_image_path=/ext/sif_image.sif
save_path=/home/gp1108/Code/VoxNet/ModelNet40Voxel.zip

url=http://modelnet.cs.princeton.edu/ModelNet40.zip
gdrive_sif_id=1H6Z9euw7WWRPUz8XGMHPtrgH8cNlt31H

export PATH="$PATH:$(python3 -m site --user-base)/bin"

# Clean up the dataset directory before downloading
rm -rf $dataset_path $dataset_archive $dataset_output

# Download the dataset
wget $url -O $dataset_archive
gdown --id $gdrive_sif_id -O $sif_image_path

# Unzip the dataset
unzip -q $dataset_archive -d /ext

# Run the script
cd $project_dir
singularity exec --bind /ext:/ext $sif_image_path python3 dataset_converter.py --input_dir $dataset_path --output_dir $dataset_output --num_workers 3

# Zip the dataset and save it
zip -r $save_path $dataset_output
