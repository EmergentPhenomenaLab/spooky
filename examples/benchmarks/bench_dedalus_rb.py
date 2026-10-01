"""Per-step cost of Dedalus (v3) on the same 2D Rayleigh-Benard problem as
``bench_rb.py``: Lx=2, Lz=1, no-slip, Pr=1, Fourier x Chebyshev, fixed dt, no
analysis output.  The problem definition is the one of
``/share/data4/jcullen/dedalus/rbc2D`` (the Dedalus RB example).

Run inside the ``dedalus3`` micromamba environment, e.g.
    OMP_NUM_THREADS=1 mpiexec -n 8 python bench_dedalus_rb.py --grid 256x129 --scheme SBDF2

Prints one line from rank 0:
    dedalus <scheme> NxxNz np=<ranks> | build <s> | <ms> ms/step
"""
import argparse
import time

import numpy as np
import dedalus.public as d3
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument('--grid', default='256x129', help='NxxNz')
parser.add_argument('--steps', type=int, default=200, help='timed steps (after warmup)')
parser.add_argument('--warmup', type=int, default=20)
parser.add_argument('--scheme', default='SBDF2', help='d3 timestepper name (SBDF2, RK222, ...)')
parser.add_argument('--dt', type=float, default=1e-3, help='fixed timestep (free-fall units)')
parser.add_argument('--ra', type=float, default=1e6)
args = parser.parse_args()

import logging
logging.getLogger().setLevel(logging.WARNING)

Nx, Nz = (int(v) for v in args.grid.lower().split('x'))
Lx, Lz = 2, 1
Rayleigh, Prandtl = args.ra, 1
dealias = 3/2
dtype = np.float64
comm = MPI.COMM_WORLD

t0 = time.perf_counter()
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

p = dist.Field(name='p', bases=(xbasis, zbasis))
b = dist.Field(name='b', bases=(xbasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1)
grad_b = d3.grad(b) + ez*lift(tau_b1)

problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
problem.add_equation("b(z=0) = Lz")
problem.add_equation("u(z=0) = 0")
problem.add_equation("b(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("integ(p) = 0")

solver = problem.build_solver(getattr(d3, args.scheme))
solver.stop_iteration = args.warmup + args.steps

b.fill_random('g', seed=42, distribution='normal', scale=1e-3)
b['g'] *= z * (Lz - z)
b['g'] += Lz - z
t_build = time.perf_counter() - t0

for _ in range(args.warmup):
    solver.step(args.dt)
comm.Barrier()
t0 = time.perf_counter()
for _ in range(args.steps):
    solver.step(args.dt)
comm.Barrier()
t_run = time.perf_counter() - t0

if comm.rank == 0:
    print(f'dedalus {args.scheme:6s} {Nx:4d}x{Nz:<4d} np={comm.size:3d} | build {t_build:5.1f} s | '
          f'{1e3*t_run/args.steps:8.3f} ms/step', flush=True)
