#!/bin/bash
# CPU comparison on one node (default c3, where the SPECTER tables of
# specter_runs/bench were measured): native RB solver (JAX CPU thread scaling,
# NumPy backend), Dedalus v3 (MPI scaling), KolmogorovFlow (NumPy vs JAX CPU).
#SBATCH -J bench-cpu
#SBATCH -o %x-%j.out
#SBATCH -N 1
#SBATCH -n 40
#SBATCH --exclusive
#SBATCH -w c3
#SBATCH -t 01:30:00

SPOOKY_ENV=${SPOOKY_ENV:-/share/data4/jcullen/envs/spooky_ad_env}
SPOOKY_DIR=${SPOOKY_DIR:-/share/data4/jcullen/spooky_AD}
B=$SPOOKY_DIR/examples/benchmarks
RB_GRIDS=128x65,256x97,256x129,512x193,512x257

# CPUs 0-39 are the 40 physical cores (even = socket 0, odd = socket 1); 40-79 are their HT siblings.
cpus() { case $1 in
    1) echo 0 ;; 4) echo 0,2,4,6 ;; 10) echo 0,2,4,6,8,10,12,14,16,18 ;;
    20) echo 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38 ;; 40) echo 0-39 ;; esac; }

hostname; lscpu | grep "Model name"; uptime

(
ml python/3.11.3
source "$SPOOKY_ENV/bin/activate"
cd "$SPOOKY_DIR"
for nt in 1 4 10 20 40; do
    echo "=== native RB, JAX CPU float64, $nt threads ==="
    extra=""; [ $nt -lt 20 ] && extra="--no-grad"
    NUMPY_BACKEND=jax taskset -c $(cpus $nt) python $B/bench_rb.py --grids $RB_GRIDS --steps 50 --evolve $extra
done
echo "=== native RB, NumPy backend (serial) ==="
NUMPY_BACKEND=numpy taskset -c 0 python $B/bench_rb.py --grids $RB_GRIDS --steps 50

echo "=== KolmogorovFlow, NumPy backend (serial) ==="
NUMPY_BACKEND=numpy taskset -c 0 python $B/bench_kolmogorov.py --grids 64,128,256,512,1024 --steps 50
for nt in 1 20 40; do
    echo "=== KolmogorovFlow, JAX CPU float64, $nt threads ==="
    NUMPY_BACKEND=jax taskset -c $(cpus $nt) python $B/bench_kolmogorov.py --grids 64,128,256,512,1024 --steps 50
done
)

(
eval "$(micromamba shell hook --shell bash)"
micromamba activate dedalus3
export OMP_NUM_THREADS=1
cd "$SPOOKY_DIR"
for scheme in SBDF2 RK222; do
    for grid in 256x103 256x129 512x257; do
        echo "=== Dedalus $scheme $grid ==="
        for np in 1 2 4 8 16 20 32 40; do
            mpiexec -n $np python $B/bench_dedalus_rb.py --grid $grid --scheme $scheme --steps 200
        done
    done
done
)
