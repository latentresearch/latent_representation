#!/bin/bash
# Experiment C (dose response): POD vs AE on the five new dose levels,
# 3 seeds each: 5 x 3 = 15 tasks. Same protocol as run_seeds.sbatch
# (500-epoch cap, early stopping with patience 20), so results are directly
# comparable with Experiment A (R = 0) and B.
#SBATCH --job-name=podae_dose
#SBATCH --partition=workq
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --array=0-14
#SBATCH --output=<DATA_ROOT>/logs/%A_%a_podae_dose.out
#SBATCH --error=<DATA_ROOT>/logs/%A_%a_podae_dose.err

module purge
export PATH=$PATH:<HOME>/.local/bin
source <DATA_ROOT>/venvs/latent_dmd/bin/activate
python3 -c "import numpy, torch" || { echo "FATAL: numpy/torch not available in $(which python3)"; exit 1; }

# ---- array index -> (dose level, seed) ------------------------------------
RADII=(250 500 1000 1500 3000)
NSEEDS=3
R=${RADII[$(( SLURM_ARRAY_TASK_ID / NSEEDS ))]}
SEED=$(( SLURM_ARRAY_TASK_ID % NSEEDS ))
DATA=<DATA_ROOT>/ldo_data_v3_r${R}
[ -d "$DATA" ] || { echo "ERROR: $DATA not found"; exit 1; }

OUTDIR=<DATA_ROOT>/logs/dose
mkdir -p "$OUTDIR"
OUT=$OUTDIR/results_r${R}_seed${SEED}.json

# ---- file-count guard: the JSON must be writable at the end ---------------
read USED FQUOTA < <(showquota | awk '$1=="/work"{print $4, $5}')
if [ -n "${FQUOTA}" ] && [ "${FQUOTA}" -gt 0 ] && [ $(( FQUOTA - USED )) -lt 10000 ]; then
  echo "FATAL: fewer than 10000 files free on /work"; exit 1
fi

if [ "${TEST:-0}" = "1" ]; then
  EXTRA="--max-samples 50 --epochs 2 --min-epochs 1"
  OUT=$OUTDIR/TEST_r${R}_seed${SEED}.json
else
  EXTRA="--max-samples 1000 --epochs 500 --min-epochs 40"
fi

echo "task=${SLURM_ARRAY_TASK_ID} exp=r${R} seed=${SEED} data=${DATA} start=$(date)"
python3 <DATA_ROOT>/pod_vs_ae_eta.py "$DATA" \
    --n-dims 8 16 32 64 --held-out-frac 0.15 $EXTRA \
    --patience 20 --batch-size 256 --num-threads 64 \
    --seed "$SEED" --out "$OUT"
echo "finished exp=r${R} seed=${SEED}: $(date) -> $OUT"
