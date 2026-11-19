#!/bin/bash
#SBATCH --job-name=svd_eps_dose
#SBATCH --partition=workq
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=<DATA_ROOT>/logs/%A_%a_svd_eps_dose.out
#SBATCH --error=<DATA_ROOT>/logs/%A_%a_svd_eps_dose.err

# One task per dose level R (Experiment C). Only Experiment-C data are analysed here.
set -euo pipefail
RADII=(250 500 1000 1500 3000)
R=${RADII[$SLURM_ARRAY_TASK_ID]}

source <ENV>/bin/activate
cd <HOME>/latent_research/analysis

python3 -I plot_singular_values.py \
    --dir-A <DATA_ROOT>/ldo_data_v3_r${R} \
    --seed 0 \
    --npz <DATA_ROOT>/logs/svd_eps_dose_r${R}.npz \
    --out <DATA_ROOT>/logs/svd_eps_dose_r${R}.pdf
