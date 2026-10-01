"""DNS Nusselt number at one Ra with the native solver, for comparison with the
SPECTER DNS reference of the paper (``specter_runs/Imgs/nu_comparison/nu_dns_gradient.csv``)
and the Dedalus runs in ``dedalus/rbc2D``.  One Ra per invocation (SLURM array).

Starts from the conductive state plus small noise and integrates ``--T`` free-fall
times in chunks of ``--chunk``, recording after each chunk three Nusselt estimates
(1 + <w theta>, <|grad T|^2>, wall flux), the kinetic energy and max |u|; the
summary averages over t > ``--tavg`` with a block-average standard error.

    NUMPY_BACKEND=jax python nu_sweep.py --ra 2e6 --grid 256x97 --dt 2e-3 --T 1500 --out results/nu_sweep

All times are free-fall units (the solver's).  The box is 2 Lz x Lz with
``--Lz`` (default pi, the SPECTER box).  KE and max |u| are reported in units
of the free-fall velocity sqrt(g alpha dT H), i.e. divided by Lz^2 and Lz,
so they do not depend on Lz and compare with ``specter_nu_series.py``.
"""
import argparse
import math
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--ra', type=float, required=True)
parser.add_argument('--grid', default='256x97')
parser.add_argument('--Lz', type=float, default=math.pi, help='layer height (box 2 Lz x Lz)')
parser.add_argument('--dt', type=float, default=2e-3, help='free-fall units')
parser.add_argument('--T', type=float, default=1500.0, help='total time, free-fall units')
parser.add_argument('--tavg', type=float, default=300.0, help='average over t > tavg')
parser.add_argument('--chunk', type=float, default=1.0, help='diagnostics interval, free-fall units')
parser.add_argument('--out', default='results/nu_sweep')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--threads', type=int, default=None, help='XLA CPU threads')
parser.add_argument('--ref-csv', default='/share/data4/jcullen/specter_runs/Imgs/nu_comparison/nu_dns_gradient.csv')
args = parser.parse_args()

if args.threads:
    os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
                               f' --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={args.threads}')
os.environ.setdefault('NUMPY_BACKEND', 'jax')

import numpy as np
from spooky._backend import JAX_ENABLED, xnp
if JAX_ENABLED:
    import jax
    jax.config.update('jax_enable_x64', True)
    print(f'jax {jax.__version__} {jax.devices()}', flush=True)
import spooky as sp
from spooky.solvers import RayleighBenard2D

Ra, Pr = args.ra, 1.0
Nx, Nz = (int(v) for v in args.grid.lower().split('x'))
os.makedirs(args.out, exist_ok=True)
tag = f'ra{Ra:.3g}_{Nx}x{Nz}'
Lz = args.Lz

n_chunk = int(round(args.chunk / args.dt))
dt = args.chunk / n_chunk                         # chunk realised exactly
grid = sp.Grid2D_wall(2.0 * Lz, Lz, Nx, Nz, dt)
solver = RayleighBenard2D(grid, ra=Ra, pr=Pr, bc='no-slip')
n_chunks = int(round(args.T / args.chunk))

rng = np.random.default_rng(args.seed)
z = np.asarray(grid.zi)
x = np.linspace(0, grid.Lx, Nx, endpoint=False)
th = np.sin(np.pi * z / Lz)[None, :] * (1e-2 * rng.standard_normal((Nx, 1))
                                    + 1e-2 * sum(np.cos(2 * np.pi * k * x / grid.Lx + rng.uniform(0, 2 * np.pi))
                                                 for k in range(1, 5))[:, None])
f0 = [np.zeros((Nx, Nz)), np.zeros((Nx, Nz)), th]


def diagnostics(fh):
    ''' [Nu_flux, Nu_grad, Nu_wall, KE, umax] from mixed-space fields (solver units;
    base temperature gradient 1, so the gradient and wall estimates need no factor). '''
    uh, wh, th = fh
    u, w, t = (grid.inverse(f) for f in fh)
    tx = grid.inverse(grid.ddx(th))
    tz = grid.inverse(solver._dz(th))
    nu_flux = 1.0 + grid.average(w * t) / solver.kappa
    nu_grad = grid.average(tx ** 2 + (tz - 1.0) ** 2)
    nu_wall = 1.0 - 0.5 * (xnp.mean(tz[:, 0]) + xnp.mean(tz[:, -1]))
    ke = 0.5 * grid.average(u ** 2 + w ** 2)
    umax = xnp.max(xnp.sqrt(u ** 2 + w ** 2))
    return xnp.stack([nu_flux, nu_grad, nu_wall, ke, umax])


ops = solver._ops_for(dt)
fh = [grid.forward(xnp.asarray(f)) for f in f0]
state = solver.init_state(fh, dt, ops=ops)
if JAX_ENABLED:
    advance = jax.jit(lambda s, n: solver._scan_steps(s, dt, n, ops=ops), static_argnums=1)
    diag = jax.jit(lambda s: diagnostics(solver.state_fields(s)))
else:
    def advance(s, n):
        for _ in range(n):
            s = solver.step_state(s, dt, ops=ops)
        return s
    diag = lambda s: diagnostics(solver.state_fields(s))

rec = np.zeros((n_chunks, 6))
t0 = time.time()
print(f'Ra={Ra:.3g} {Nx}x{Nz} dt={args.dt:g} ff, {n_chunks} chunks of {n_chunk} steps', flush=True)
for i in range(n_chunks):
    state = advance(state, n_chunk - (solver.n_startup if i == 0 else 0))
    d = np.asarray(diag(state))
    rec[i] = [(i + 1) * args.chunk, d[0], d[1], d[2], d[3] / Lz ** 2, d[4] / Lz]
    if not np.isfinite(d).all():
        print(f'blow-up at t={rec[i, 0]:.1f}', flush=True)
        rec = rec[:i + 1]
        break
    if (i + 1) % 50 == 0 or i == 0:
        cfl = d[4] * dt / (grid.Lx / Nx)
        print(f't={rec[i, 0]:8.1f}  Nu_flux={d[0]:8.4f} Nu_grad={d[1]:8.4f} Nu_wall={d[2]:8.4f}  '
              f'KE={rec[i, 4]:.5f}  CFL={cfl:.3f}  ({(time.time()-t0)/(i+1)*1e3/n_chunk:.2f} ms/step)', flush=True)
        np.savez(os.path.join(args.out, f'{tag}.npz'), t=rec[:i + 1, 0], nu_flux=rec[:i + 1, 1],
                 nu_grad=rec[:i + 1, 2], nu_wall=rec[:i + 1, 3], ke=rec[:i + 1, 4], umax=rec[:i + 1, 5],
                 Ra=Ra, dt_ff=dt, Lz=Lz, Nx=Nx, Nz=Nz)
np.savez(os.path.join(args.out, f'{tag}.npz'), t=rec[:, 0], nu_flux=rec[:, 1], nu_grad=rec[:, 2],
         nu_wall=rec[:, 3], ke=rec[:, 4], umax=rec[:, 5], Ra=Ra, dt_ff=dt, Lz=Lz, Nx=Nx, Nz=Nz)


def block_se(v, nb=10):
    blocks = np.array_split(v, nb)
    means = np.array([b.mean() for b in blocks])
    return means.std(ddof=1) / np.sqrt(nb)


m = rec[:, 0] > args.tavg
ref = ''
if os.path.exists(args.ref_csv):
    rows = np.genfromtxt(args.ref_csv, delimiter=',', skip_header=1, usecols=(0, 1, 2))
    hit = rows[np.isclose(rows[:, 0], Ra, rtol=1e-6)]
    if len(hit):
        ref = f'  | SPECTER DNS ref (gradient): {hit[0, 1]:.4f} +- {hit[0, 2]:.4f} (std)'
line = (f'SUMMARY Ra={Ra:.3g} {Nx}x{Nz} dt={args.dt:g} t in ({args.tavg:g}, {rec[-1, 0]:g}): '
        f'Nu_flux={rec[m, 1].mean():.4f}+-{block_se(rec[m, 1]):.4f} (std {rec[m, 1].std():.3f})  '
        f'Nu_grad={rec[m, 2].mean():.4f}+-{block_se(rec[m, 2]):.4f}  Nu_wall={rec[m, 3].mean():.4f}+-{block_se(rec[m, 3]):.4f}  '
        f'KE={rec[m, 4].mean():.5f}  umax={rec[m, 5].max():.3f}  [{(time.time()-t0)/60:.1f} min]{ref}')
print(line, flush=True)
with open(os.path.join(args.out, 'summary.txt'), 'a') as f:
    print(line, file=f)
