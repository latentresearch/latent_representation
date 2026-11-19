#!/bin/bash
# Seeds 0-4 for both experiments: 2 x 5 = 10 tasks.
# Epoch cap raised from 150 to 500 so that early stopping (patience 20),
# not the cap, ends AE training. Seed 0 is rerun for consistency.
# Settings copied from ch3_full_sweep.sbatch.
#SBATCH --job-name=podae_seeds
#SBATCH --partition=workq
#SBATCH --account=loni_latent_pde
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00            # <- adjust after checking the old sweep's Elapsed time
#SBATCH --array=0-9
#SBATCH --output=<DATA_ROOT>/logs/%A_%a_podae_seeds.out
#SBATCH --error=<DATA_ROOT>/logs/%A_%a_podae_seeds.err

module purge
export PATH=$PATH:<HOME>/.local/bin

# Python environment with numpy/pandas/torch (activated inside the job,
# so the job does not depend on the state of the submitting terminal)
source <DATA_ROOT>/venvs/latent_dmd/bin/activate
python3 -c "import numpy" || { echo "FATAL: numpy not available in $(which python3)"; exit 1; }

# ---- array index -> (experiment, seed) ----------------------------------
EXPS=(A B)
EXP=${EXPS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
SEED=$(( SLURM_ARRAY_TASK_ID % 5 ))              # seeds 0,1,2,3,4

if [ "$EXP" = "A" ]; then
  DATA=<DATA_ROOT>/ldo_data_v3_fixedwell          # CHECK against ch3_fixedwell_sweep.sbatch
else
  DATA=<DATA_ROOT>/ldo_data_v3_reconfirm          # same as ch3_full_sweep.sbatch
fi

if [ ! -d "$DATA" ]; then
  echo "ERROR: data directory $DATA not found"; exit 1
fi

OUTDIR=<DATA_ROOT>/logs/seeds
mkdir -p "$OUTDIR"
OUT=$OUTDIR/results_${EXP}_seed${SEED}.json

# ---- full run, or a quick test with: sbatch --export=ALL,TEST=1 ... -----
if [ "${TEST:-0}" = "1" ]; then
  EXTRA="--max-samples 50 --epochs 2 --min-epochs 1"
  OUT=$OUTDIR/TEST_${EXP}_seed${SEED}.json
else
  EXTRA="--max-samples 1000 --epochs 500 --min-epochs 40"
fi

echo "task=${SLURM_ARRAY_TASK_ID} exp=${EXP} seed=${SEED} data=${DATA} start=$(date)"

python3 <DATA_ROOT>/pod_vs_ae_eta.py "$DATA" \
    --n-dims 8 16 32 64 --held-out-frac 0.15 $EXTRA \
    --patience 20 --batch-size 256 --num-threads 64 \
    --seed "$SEED" --out "$OUT"

echo "finished exp=${EXP} seed=${SEED}: $(date) -> $OUT"
