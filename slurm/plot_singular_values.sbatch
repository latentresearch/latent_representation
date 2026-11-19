#!/bin/bash
#SBATCH --job-name=svd_eps
#SBATCH --partition=workq
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=<DATA_ROOT>/logs/%j_svd_eps.out
#SBATCH --error=<DATA_ROOT>/logs/%j_svd_eps.err

set -euo pipefail
source <ENV>/bin/activate
cd <HOME>/latent_research/analysis

python3 -I plot_singular_values.py \
    --dir-A <DATA_ROOT>/ldo_data_v3_fixedwell \
    --dir-B <DATA_ROOT>/ldo_data_v3_reconfirm \
    --seed 0 \
    --out singular_values.pdf
