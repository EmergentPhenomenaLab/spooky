#!/bin/bash
# GPU benchmarks on one L4 (default c6; pass `-w c4` to sbatch for an A100):
# native RB solver per-step cost, KolmogorovFlow per-step cost, and the
# dt/Nz study on the converted SPECTER PO1.
#SBATCH -J bench-gpu
#SBATCH -o %x-%j.out
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -w c6
#SBATCH -t 02:00:00

SPOOKY_ENV=${SPOOKY_ENV:-/share/data4/jcullen/envs/spooky_jax_gpu}
SPOOKY_DIR=${SPOOKY_DIR:-/share/data4/jcullen/spooky_AD}
B=$SPOOKY_DIR/examples/benchmarks

ml python/3.11.3
source "$SPOOKY_ENV/bin/activate"
cd "$SPOOKY_DIR"
export NUMPY_BACKEND=jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false

hostname
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

echo "=== native RB float64 ==="
python $B/bench_rb.py --grids 128x65,256x97,256x129,512x193,512x257 --steps 200 --mem --evolve
echo "=== native RB float32 ==="
python $B/bench_rb.py --grids 256x129,512x257 --steps 200 --dtype float32

echo "=== KolmogorovFlow float64 ==="
python $B/bench_kolmogorov.py --grids 64,128,256,512,1024,2048 --steps 200
echo "=== KolmogorovFlow float32 ==="
python $B/bench_kolmogorov.py --grids 256,1024,2048 --steps 200 --dtype float32

echo "=== PO1 dt / Nz study (float64) ==="
python $SPOOKY_DIR/examples/rayleigh_benard/specter_orbit_dt_study.py
