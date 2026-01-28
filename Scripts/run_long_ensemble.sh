#!/bin/bash
#SBATCH --partition=Odyssey               # Partition name
#SBATCH --gres=gpu:a100:1             # GPU request
#SBATCH --job-name=Samudra-Long_ENS         # Job name

# Define the list of IDs
IDS=(
  # control
  sigma_0.5
  sigma_1.0
  sigma_2.0
  # sigma_5.0 # Unstable on some runs
)

# Loop over IDs and run the command
for ID in "${IDS[@]}"; do
  echo "Running ensemble: ${ID}"
  srun python src/noisy_ensemble.py --config "./configs/Long-ENS/ENS-${ID}.yaml"
done
