#!/bin/bash
#SBATCH -J insar_train
#SBATCH -o insar_train.o%j
#SBATCH -e insar_train.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A <YOUR_ALLOCATION>
#SBATCH --mail-type=all
#SBATCH --mail-user=<YOUR_EMAIL>

# 1. Move to your working directory
cd /scratch/<YOUR_SCRATCH_DIR>/exp-insar

# 2. Load necessary system modules
module load gcc cuda

# 3. Initialize Conda for this shell session
source $(conda info --base)/etc/profile.d/conda.sh

# 4. Activate your environment
conda activate gpu_env

# 5. Execute your training script
python3 train_vanilla_unet.py
