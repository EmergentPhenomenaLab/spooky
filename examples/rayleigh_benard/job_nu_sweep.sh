#!/bin/bash
# DNS Nu(Ra) sweep with the native solver, one Ra per array task (CPU, 4 XLA threads:
# ~1.8 h for 1500 free-fall times at 256x97; GPU: minutes).
#SBATCH -J nu-sweep
#SBATCH -o results/%x-%A_%a.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH --array=0-9

SPOOKY_ENV=${SPOOKY_ENV:-/share/data4/jcullen/envs/spooky_ad_env}
GRID=${GRID:-256x97}
RAS=(1e5 2e5 3e5 4e5 9e5 1e6 2e6 5e6 1e7 2e7)
RA=${RAS[$SLURM_ARRAY_TASK_ID]}
# steady below Ra ~ 4.5e5: shorter runs; smaller dt at the top of the range (advective CFL)
case $RA in
    1e5|2e5|3e5|4e5) T=600;  DT=2e-3 ;;
    1e7|2e7)         T=1500; DT=1.5e-3 ;;
    *)               T=1500; DT=2e-3 ;;
esac

ml python/3.11.3
source "$SPOOKY_ENV/bin/activate"
cd /share/data4/jcullen/spooky_AD/examples/rayleigh_benard
export NUMPY_BACKEND=jax XLA_PYTHON_CLIENT_PREALLOCATE=false
hostname; nproc
python nu_sweep.py --ra $RA --grid $GRID --dt $DT --T $T --tavg 300 --threads 4 --out results/nu_sweep_$GRID
