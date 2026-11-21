#!/bin/bash
# Re-run the 18 AE fits that hit the 500-epoch cap, with a 1500-epoch cap.
# Array task -> (experiment, seed, latent sizes that hit the cap).
# Same seed and settings as the 500-epoch sweep, so the first 500 epochs should reproduce.
#
# Replace <ACCOUNT>, <DATA_ROOT>, <HOME> before submitting.
#SBATCH --job-name=podae_cap1500
#SBATCH --partition=workq
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=<DATA_ROOT>/logs/%A_%a_cap1500.out
#SBATCH --error=<DATA_ROOT>/logs/%A_%a_cap1500.err

module purge
source <DATA_ROOT>/venvs/latent_dmd/bin/activate
python3 -c "import numpy" || { echo "FATAL: numpy not available"; exit 1; }

case $SLURM_ARRAY_TASK_ID in
  0) EXP=A; SEED=0; NS="16" ;;
  1) EXP=A; SEED=1; NS="8 16 32 64" ;;
  2) EXP=A; SEED=2; NS="64" ;;
  3) EXP=A; SEED=3; NS="64" ;;
  4) EXP=A; SEED=4; NS="16" ;;
  5) EXP=B; SEED=0; NS="8 16" ;;
  6) EXP=B; SEED=1; NS="8 16" ;;
  7) EXP=B; SEED=2; NS="8 16" ;;
  8) EXP=B; SEED=3; NS="8 16 32" ;;
  9) EXP=B; SEED=4; NS="8" ;;
esac

if [ "$EXP" = "A" ]; then DATA=<DATA_ROOT>/ldo_data_v3_fixedwell
else DATA=<DATA_ROOT>/ldo_data_v3_reconfirm; fi
[ -d "$DATA" ] || { echo "ERROR: $DATA not found"; exit 1; }

OUTDIR=<DATA_ROOT>/logs/cap1500
mkdir -p "$OUTDIR"
OUT=$OUTDIR/results_${EXP}_seed${SEED}_cap1500.json
EXTRA="--max-samples 1000 --epochs 1500 --min-epochs 40"
PREFIX=""

# Quick check before the full run: sbatch --export=ALL,TEST=1 --array=0 run_capped_1500.sbatch
if [ "${TEST:-0}" = "1" ]; then
  EXTRA="--max-samples 50 --epochs 3 --min-epochs 1"
  OUT=$OUTDIR/TEST_${EXP}_seed${SEED}_cap1500.json
fi

echo "task=${SLURM_ARRAY_TASK_ID} exp=${EXP} seed=${SEED} n=[${NS}] start=$(date)"

python3 <DATA_ROOT>/pod_vs_ae_eta.py "$DATA" \
    --n-dims $NS --held-out-frac 0.15 $EXTRA \
    --patience 20 --batch-size 256 --num-threads 64 \
    --seed "$SEED" --out "$OUT"

echo "finished exp=${EXP} seed=${SEED}: $(date) -> $OUT"
