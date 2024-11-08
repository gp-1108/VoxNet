#!/bin/bash

#SBATCH --job-name voxel_g
#SBATCH --error error_%j.txt
#SBATCH --output output_%j.txt
#SBATCH --mail-user pietro.girotto@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 24:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 20G

cd $WORKING_DIR/Code/voxnet
echo $PWD

bash ./script.sh