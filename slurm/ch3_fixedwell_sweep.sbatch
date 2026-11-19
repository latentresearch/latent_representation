#!/bin/bash
#SBATCH --job-name=ch3_fixedwell_sweep
#SBATCH --partition=workq
#SBATCH --account=loni_latent_pde
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=<DATA_ROOT>/logs/%j_ch3_fixedwell_sweep.out
#SBATCH --error=<DATA_ROOT>/logs/%j_ch3_fixedwell_sweep.err

module purge
export PATH=$PATH:<HOME>/.local/bin

# Same n values, same settings, same seed as the confirmed variable-well
# sweep (ch3_full_sweep.sbatch) -- only the dataset differs, so any change
# in the POD/AE gap is attributable to removing well-location translation,
# not to a settings mismatch.
python3 <DATA_ROOT>/pod_vs_ae_eta.py <DATA_ROOT>/ldo_data_v3_fixedwell \
    --n-dims 8 16 32 64 --held-out-frac 0.15 --max-samples 1000 \
    --epochs 150 --min-epochs 40 --patience 20 --batch-size 256 \
    --num-threads 64 --seed 0 \
    --out <DATA_ROOT>/logs/ch3_fixedwell_sweep_results.json

echo "========================================"
echo "Chapter 3 fixed-well sweep finished: $(date)"
echo "========================================"
