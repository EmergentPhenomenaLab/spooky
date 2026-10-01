"""Timestep / resolution study on a converged SPECTER periodic orbit.

Converts the SPECTER PO1 (see ``specter_orbit_check.py``) to the native grid and
integrates one period with the jitted ``ad_evolve`` for several ``dt`` and ``Nz``.
The recurrence residual ||Phi_T(U)-U||/||U|| measures how far the native discrete
flow is from SPECTER's orbit: it plateaus at the interpolation + SPECTER
discretisation error for small ``dt`` and grows as dt^2 (SBDF2) once the native
temporal error dominates; NaN means the explicit advection went unstable.  This
sets the usable ``dt``, i.e. the cost per simulated time unit, which is what has
to be compared with SPECTER's production ``dt`` (1e-3 at 512x512).

``dt`` is adjusted to divide the period exactly (the IMEX path realises T with
round(T/dt) steps), so every run integrates exactly one period.  Free-fall units
in the SPECTER box: T and dt are SPECTER's.

    NUMPY_BACKEND=jax python specter_orbit_dt_study.py --grids 256x129 --dts 1e-3,2e-3,4e-3
"""
import argparse
import os
import time

os.environ.setdefault('NUMPY_BACKEND', 'jax')

parser = argparse.ArgumentParser()
parser.add_argument('--grids', default='256x97,256x129,256x193')
parser.add_argument('--dts', default='5e-4,1e-3,2e-3,4e-3,8e-3,1.6e-2',
                    help='target timesteps (free-fall units)')
args = parser.parse_args()

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

import spooky as sp
from spooky import rb_convert as rc
from spooky.solvers import RayleighBenard2D

print(f'jax {jax.__version__}  devices {jax.devices()}')

d = '/share/data4/jcullen/RB/arclength/PO1_up/converged/'
Ra, Pr = 19977416.43527502, 1.0
T = 10.831331173674736
fs = rc.load_specter_fields(d, 2, (512, 512))
print(f'PO1  Ra={Ra:.6e}  T={T:.6f} (free-fall)  SPECTER Nu={rc.specter_nusselt(fs, Ra, Pr):.5f}')


def norm(grid, fields):
    return np.sqrt(sum(float(grid.average(f ** 2)) for f in fields))


print(f'{"grid":>9s} {"dt_ff":>10s} {"steps":>7s} {"residual":>11s} {"Nu(T)":>9s} {"KE ratio":>9s} '
      f'{"build s":>8s} {"run s":>7s} {"ms/step":>8s}')
for g in args.grids.split(','):
    Nx, Nz = (int(v) for v in g.lower().split('x'))
    for dt_target in (float(v) for v in args.dts.split(',')):
        Nt = int(round(T / dt_target))
        dt = T / Nt
        t0 = time.perf_counter()
        grid = sp.Grid2D_wall(2 * np.pi, np.pi, Nx, Nz, dt)
        solver = RayleighBenard2D(grid, ra=Ra, pr=Pr, bc='no-slip')
        t_build = time.perf_counter() - t0
        assert solver._nsteps(T) == Nt
        fd = [jnp.array(f) for f in rc.specter_to_native(fs, grid)]
        ke0 = 0.5 * float(grid.average(fd[0] ** 2 + fd[1] ** 2))

        fwd = jax.jit(lambda f: solver.ad_evolve(f, T))
        t0 = time.perf_counter()
        out = fwd(fd); jax.block_until_ready(out)
        t_run = time.perf_counter() - t0                 # includes compilation
        res = norm(grid, [a - b for a, b in zip(out, fd)]) / norm(grid, fd)
        ke = 0.5 * float(grid.average(out[0] ** 2 + out[1] ** 2))
        print(f'{g:>9s} {T / Nt:10.3e} {Nt:7d} {res:11.3e} {float(solver.nusselt(out)):9.4f} '
              f'{ke / ke0:9.5f} {t_build:8.1f} {t_run:7.1f} {1e3 * t_run / Nt:8.3f}', flush=True)
