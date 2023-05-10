#!/bin/bash

#SBATCH --job-name=Unet.py
#SBATCH --output=testUnet.out
#SBATCH --error=testUnet.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

singularity exec --nv --bind /scratch/$USER --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro /scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/ML
python Unet.py
"
