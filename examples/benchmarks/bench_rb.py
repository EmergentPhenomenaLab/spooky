"""Per-step cost and reverse-mode memory of the native RB solver.

Times one jitted forward step, one JVP step and one gradient (VJP) step of
``RayleighBenard2D.ad_evolve`` at a few resolutions, and (with ``--mem``)
reports the compiled temporary memory of each program, from which the
reverse-mode tape per step follows.  Same problem as ``bench_dedalus_rb.py``:
box 2 x 1, no-slip, Pr = 1, dt = 1e-3 free-fall units.  Requires the JAX
backend (NumPy: ``solver.evolve`` only).

Examples
--------
    NUMPY_BACKEND=jax python bench_rb.py                          # CPU or GPU, float64
    NUMPY_BACKEND=jax python bench_rb.py --dtype float32 --steps 200
    NUMPY_BACKEND=jax python bench_rb.py --grids 256x129 --mem
    NUMPY_BACKEND=numpy python bench_rb.py                        # solver.evolve only

Reference numbers for the JVP (2026-10-01, real-valued operators, Ra = 2e7,
SPECTER box; ``/share/data4/jcullen/rb_bench/3b``): A100 0.43 ms/step at 256x97,
1.65 at 512x257; L4 1.11 and 12.2; JAX CPU (c3, 10 cores) 25 at 256x97 and 142
at 512x193.  Reverse-mode tape (2026-09-09, before that change): ~2.2 MB/step at
256x129, ~8.8 MB/step at 512x257.
"""
import argparse
import os
import time

os.environ.setdefault('NUMPY_BACKEND', 'jax')

parser = argparse.ArgumentParser()
parser.add_argument('--grids', default='128x65,256x129,512x257',
                    help='comma-separated NxxNz list')
parser.add_argument('--steps', type=int, default=100, help='timesteps per timing')
parser.add_argument('--dtype', default='float64', choices=['float64', 'float32'])
parser.add_argument('--mem', action='store_true',
                    help='also report compiled temp memory (tape) per program')
parser.add_argument('--ra', type=float, default=1e6)
parser.add_argument('--repeat', type=int, default=3,
                    help='timed runs per program after the compile run; the minimum is reported')
parser.add_argument('--evolve', action='store_true',
                    help='also time solver.evolve (per-step jitted Python loop, the path Newton uses)')
parser.add_argument('--no-grad', action='store_true', help='skip the reverse-mode timing')
args = parser.parse_args()

import numpy as np

if os.environ['NUMPY_BACKEND'] != 'jax':
    # NumPy backend: only solver.evolve can be timed (no ad_evolve, no AD).
    import spooky as sp
    from spooky.solvers import RayleighBenard2D
    print(f'numpy {np.__version__}')
    for g in args.grids.split(','):
        Nx, Nz = (int(v) for v in g.lower().split('x'))
        grid = sp.Grid2D_wall(2.0, 1.0, Nx, Nz, 1e-3)
        solver = RayleighBenard2D(grid, ra=args.ra, pr=1.0, bc='no-slip')
        z = np.asarray(grid.zi)
        th = 1e-2 * np.sin(np.pi * z)[None, :] * np.random.default_rng(0).standard_normal((Nx, 1))
        f0 = [np.zeros((Nx, Nz)), np.zeros((Nx, Nz)), th]
        best = float('inf')
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            solver.evolve([f.copy() for f in f0], args.steps * grid.dt, write_outputs=False)
            best = min(best, time.perf_counter() - t0)
        print(f'{Nx:4d}x{Nz:<4d} | evolve {1e3*best/args.steps:7.2f} ms/step', flush=True)
    raise SystemExit

import jax
jax.config.update('jax_enable_x64', args.dtype == 'float64')
import jax.numpy as jnp

import spooky as sp
from spooky.solvers import RayleighBenard2D

print(f'jax {jax.__version__}  devices {jax.devices()}  dtype {args.dtype}')


def timed(fn, *a):
    out = fn(*a); jax.block_until_ready(out)          # compile + first run
    best = float('inf')
    for _ in range(args.repeat):
        t0 = time.perf_counter(); out = fn(*a); jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    return best


def bench(Nx, Nz, nsteps):
    grid = sp.Grid2D_wall(2.0, 1.0, Nx, Nz, 1e-3)
    t0 = time.perf_counter()
    solver = RayleighBenard2D(grid, ra=args.ra, pr=1.0, bc='no-slip')
    t_build = time.perf_counter() - t0
    ops = [solver.SLp, solver.SProj] + [v for o in solver._ops_for(grid.dt).values() for v in o.values()]
    op_bytes = sum(int(np.asarray(v).nbytes) for v in ops)

    z = np.asarray(grid.zi)
    rng = np.random.default_rng(0)
    th = 1e-2 * np.sin(np.pi * z)[None, :] * rng.standard_normal((Nx, 1))
    f0 = [jnp.zeros((Nx, Nz)), jnp.zeros((Nx, Nz)), jnp.array(th)]
    d0 = [jnp.array(rng.standard_normal((Nx, Nz))) for _ in range(3)]
    T = nsteps * grid.dt

    fwd = jax.jit(lambda f: solver.ad_evolve(f, T))
    jvp = jax.jit(lambda f, d: jax.jvp(lambda g: solver.ad_evolve(g, T), (f,), (d,))[1])
    grad = jax.jit(jax.grad(lambda f: sum(jnp.sum(o ** 2) for o in solver.ad_evolve(f, T))))

    t_f = timed(fwd, f0); t_j = timed(jvp, f0, d0)
    t_g = float('nan') if args.no_grad else timed(grad, f0)
    ms = 1e3 / nsteps
    line = (f'{Nx:4d}x{Nz:<4d} | build {t_build:5.1f} s | ops {op_bytes/1e9:5.2f} GB | '
            f'fwd {ms*t_f:7.2f} ms/step | jvp {ms*t_j:7.2f} | grad {ms*t_g:7.2f}')
    if args.evolve:
        t_e = timed(lambda f: solver.evolve(f, T, write_outputs=False), f0)
        line += f' | evolve {ms*t_e:7.2f}'
    print(line, flush=True)
    if args.mem and not args.no_grad:
        mf = fwd.lower(f0).compile().memory_analysis().temp_size_in_bytes
        mj = jvp.lower(f0, d0).compile().memory_analysis().temp_size_in_bytes
        mg = grad.lower(f0).compile().memory_analysis().temp_size_in_bytes
        print(f'           temp: fwd {mf/1e6:8.1f} MB | jvp {mj/1e6:8.1f} MB | '
              f'grad {mg/1e6:8.1f} MB  -> tape ~{mg/1e6/nsteps:6.2f} MB/step', flush=True)


for g in args.grids.split(','):
    Nx, Nz = (int(v) for v in g.lower().split('x'))
    bench(Nx, Nz, args.steps)
