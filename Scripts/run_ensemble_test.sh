#!/usr/bin/env bash

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
  python src/noisy_ensemble.py --config "./configs/ENS/ENS-${ID}.yaml"
done
