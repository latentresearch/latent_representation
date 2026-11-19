#!/bin/bash
# Dose-response ensembles (Ch.3, Experiment C): well offset drawn uniformly from
# the 250 m corner lattice within +/- R of the domain centre.
# Submit one job per dose level, e.g.
#   sbatch --export=ALL,R=250  darcyFoam_v3_radius.sbatch
#   sbatch --export=ALL,R=500  darcyFoam_v3_radius.sbatch   ... (R = 1000, 1500, 3000)
#SBATCH --job-name=ldo_v3_radius
#SBATCH --partition=workq
#SBATCH --account=loni_latent_pde
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --output=<DATA_ROOT>/logs/%j_v3_radius.out
#SBATCH --error=<DATA_ROOT>/logs/%j_v3_radius.err

module purge
module load openfoam
export PATH=$PATH:<HOME>/.local/bin:<HOME>/OpenFOAM/jing-v2212/platforms/linux64GccDPInt32Opt/bin

# Python environment with numpy/pandas/torch (activated inside the job,
# so the job does not depend on the state of the submitting terminal)
source <DATA_ROOT>/venvs/latent_dmd/bin/activate
python3 -c "import numpy" || { echo "FATAL: numpy not available in $(which python3)"; exit 1; }

R=${R:?"set R, e.g. sbatch --export=ALL,R=500 darcyFoam_v3_radius.sbatch"}
WORK_DIR=<DATA_ROOT>
BASE_CASE=<DATA_ROOT>/base_case
ENSEMBLE_DIR=<DATA_ROOT>/ensemble_ch3_r${R}
DATA_DIR=<DATA_ROOT>/ldo_data_v3_r${R}
LOG_DIR=<DATA_ROOT>/logs
mkdir -p ${DATA_DIR} ${LOG_DIR}
echo "Dose level R=${R} m -> ${ENSEMBLE_DIR}, ${DATA_DIR}"

# ---- file-count guard: refuse to start if fewer than 300k files remain ----
read USED FQUOTA < <(showquota | awk '$1=="/work"{print $4, $5}')
if [ -n "${FQUOTA}" ] && [ "${FQUOTA}" -gt 0 ]; then
    FREE=$(( FQUOTA - USED ))
    echo "/work files: ${USED} used of ${FQUOTA} (${FREE} free)"
    if [ ${FREE} -lt 300000 ]; then
        echo "FATAL: fewer than 300000 files free on /work -- clean up before running."
        exit 1
    fi
fi

N_SAMPLES=${N_SAMPLES:-1000}
LAST_SAMPLE_ID=$(printf "%04d" $((N_SAMPLES - 1)))
if [ ! -d "${ENSEMBLE_DIR}/sample_${LAST_SAMPLE_ID}" ]; then
    echo "Generating ${N_SAMPLES} cases with R=${R} m"
    # --seed 0 matches the fixed-well run, so rock fields are identical sample by sample
    python3 ${WORK_DIR}/generate_samples_v3_radius.py \
        --n ${N_SAMPLES} --base ${BASE_CASE} --out ${ENSEMBLE_DIR} --seed 0 --radius ${R}
    if [ $? -ne 0 ]; then
        echo "FATAL: generate_samples_v3_radius.py failed -- aborting before any solves."
        exit 1
    fi
else
    echo "Ensemble already generated (found sample_${LAST_SAMPLE_ID}) -- skipping generation."
fi

START=${START:-0}
END=${END:-$((N_SAMPLES - 1))}

run_case() {
    i=$1
    SAMPLE_ID=$(printf "%04d" ${i})
    CASE_DIR=${ENSEMBLE_DIR}/sample_${SAMPLE_ID}
    NPY=${DATA_DIR}/sample_${SAMPLE_ID}_p.npy
    if [ ! -d "${CASE_DIR}" ]; then echo "ERROR: ${CASE_DIR} does not exist"; return; fi
    if [ -f "${NPY}" ]; then echo "SKIP sample_${SAMPLE_ID}"; return; fi
    echo "START sample_${SAMPLE_ID}"
    cd ${CASE_DIR}
    blockMesh >> solver.log 2>&1 || { echo "ERROR: blockMesh failed for sample_${SAMPLE_ID}"; touch FAILED_MESH; return; }
    topoSet   >> solver.log 2>&1 || { echo "ERROR: topoSet failed for sample_${SAMPLE_ID}"; touch FAILED_MESH; return; }
    darcyFoam >> solver.log 2>&1 || { echo "ERROR: solver failed for sample_${SAMPLE_ID}"; touch FAILED_SOLVER; return; }
    if ! grep -q "^End$" solver.log; then
        echo "ERROR: no End for sample_${SAMPLE_ID}"; touch FAILED_SOLVER; return
    fi
    python3 ${WORK_DIR}/extract_fields_v3.py \
        --case ${CASE_DIR} --sample ${i} --out ${DATA_DIR} \
        > ${CASE_DIR}/extract.log 2>&1 || {
            echo "ERROR: extraction failed for sample_${SAMPLE_ID}"; touch FAILED_EXTRACT; return; }
    # Keep inputs (0/, constant/, system/) and logs; delete the solved time
    # directories, whose fields are now stored in ${DATA_DIR} as .npy.
    # Any case can be re-solved from its inputs if needed.
    find ${CASE_DIR} -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
        -regex '.*/[0-9]+(\.[0-9]+)?(e[+-]?[0-9]+)?' ! -name 0 -exec rm -rf {} +
    rm -rf ${CASE_DIR}/processor*
    echo "DONE sample_${SAMPLE_ID}"
}
export -f run_case
export WORK_DIR ENSEMBLE_DIR DATA_DIR PATH

MAX_JOBS=${SLURM_NTASKS:-64}
echo "Parallel workers: ${MAX_JOBS}   Range: sample_${START} .. sample_${END}"
for ((i=START; i<=END; i++)); do
    run_case $i &
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do sleep 10; done
done
wait

echo "========================================"
echo "Radius R=${R} batch ${START}-${END} finished: $(date)"
n_done=$(ls ${DATA_DIR}/sample_*_p.npy 2>/dev/null | wc -l)
n_fail_mesh=$(ls ${ENSEMBLE_DIR}/sample_*/FAILED_MESH 2>/dev/null | wc -l)
n_fail_solve=$(ls ${ENSEMBLE_DIR}/sample_*/FAILED_SOLVER 2>/dev/null | wc -l)
n_fail_extract=$(ls ${ENSEMBLE_DIR}/sample_*/FAILED_EXTRACT 2>/dev/null | wc -l)
echo "Extracted: ${n_done}   Failed mesh: ${n_fail_mesh}   Failed solve: ${n_fail_solve}   Failed extract: ${n_fail_extract}"
echo "========================================"
