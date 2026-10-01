#!/bin/bash
# Native Newton re-convergence + Floquet spectrum of the SPECTER PO1 at Ra = 2e7
# (CPU: ~1 h per Newton iteration at 256x97 with 4 XLA threads; GPU: minutes).
#SBATCH -J newton-po1
#SBATCH -o results/%x-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 36:00:00

SPOOKY_ENV=${SPOOKY_ENV:-/share/data4/jcullen/envs/spooky_ad_env}
GRID=${GRID:-256x97}
DT=${DT:-2e-3}

ml python/3.11.3
source "$SPOOKY_ENV/bin/activate"
cd /share/data4/jcullen/spooky_AD/examples/rayleigh_benard
export NUMPY_BACKEND=jax XLA_PYTHON_CLIENT_PREALLOCATE=false
hostname; nproc
python newton_po1.py --grid $GRID --dt $DT --threads 4 --out results/newton_po1_${GRID}_dt${DT}
