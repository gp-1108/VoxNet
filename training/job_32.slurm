#!/bin/bash

#SBATCH --job-name voxel_g
#SBATCH --error error_%j.txt
#SBATCH --output output_%j.txt
#SBATCH --mail-user pietro.girotto@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 4-12:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 170G
#SBATCH --gres=gpu:rtx

cd $WORKING_DIR/Code/VoxNet/training
echo $PWD

bash ./script_32.sh
