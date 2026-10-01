#!/bin/bash
# Fixed-Ra Newton + Floquet on a SPECTER PO1 away from the fold (Ra = 1.7e7 / 1.9e7).
# Usage: RA=1.7e7 REF=/share/data4/jcullen/RB/ra1.7e7/512-512/1721_ra1.6e7 sbatch [-w c4 --gres=gpu:1] job_newton_po1_ra.sh
#SBATCH -J newton-po1-ra
#SBATCH -o results/%x-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 30:00:00
RA=${RA:-1.7e7}; REF=${REF:-/share/data4/jcullen/RB/ra1.7e7/512-512/1721_ra1.6e7}
GRID=${GRID:-256x97}; DT=${DT:-2e-3}
ml python/3.11.3
if [ -n "$CUDA_VISIBLE_DEVICES" ] || nvidia-smi >/dev/null 2>&1 && [ -n "$SLURM_JOB_GPUS$SLURM_STEP_GPUS" ]; then
  source /share/data4/jcullen/envs/spooky_jax_gpu/bin/activate; THREADS=""
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | sort -t, -k2,2n -k3,3n | head -1 | cut -d, -f1)
else
  source /share/data4/jcullen/envs/spooky_ad_env/bin/activate; THREADS="--threads 4"
fi
cd /share/data4/jcullen/spooky_AD/examples/rayleigh_benard
export NUMPY_BACKEND=jax XLA_PYTHON_CLIENT_PREALLOCATE=false
hostname; echo "RA=$RA REF=$REF CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
python newton_po1.py --ref $REF --ra $RA --grid $GRID --dt $DT $THREADS --resume --out results/newton_po1_ra${RA}_${GRID}_dt${DT}
