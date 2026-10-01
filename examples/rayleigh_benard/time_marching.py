'''
2D Rayleigh-Benard convection (periodic x, wall-bounded z), free-fall units.

Evolves a small perturbation of the conduction state and plots the final
temperature field. Run with the default NumPy backend, or set
NUMPY_BACKEND=jax to use the differentiable path.  At Ra = 1e5 the flow settles
into two steady rolls with Nu = 4.994 (t ~ 100).
'''
import numpy as np
import matplotlib.pyplot as plt

import spooky as sp
from spooky.solvers import RayleighBenard2D

import params as pm

grid = sp.Grid2D_wall(Lx=pm.Lx, Lz=pm.Lz, Nx=pm.Nx, Nz=pm.Nz, dt=pm.dt)
solver = RayleighBenard2D(grid, ra=pm.Ra, pr=pm.Pr, bc=pm.bc)

# Initial condition: random small temperature perturbation that vanishes at the
# walls (theta = 0 at z = 0, Lz). Velocity starts from rest.
z = np.asarray(grid.zi)
x = np.linspace(0, pm.Lx, pm.Nx, endpoint=False)[:, None]
rng = np.random.default_rng(0)
th = 1e-2 * np.sin(np.pi * z / pm.Lz)[None, :] * rng.standard_normal((pm.Nx, 1))
uu = np.zeros((pm.Nx, pm.Nz))
ww = np.zeros((pm.Nx, pm.Nz))
fields = [uu, ww, th]

# Evolve, writing kinetic energy and Nusselt number every bstep steps
fields = solver.evolve(fields, T=pm.Tevolve, bstep=pm.bstep, bpath='.', ostep=pm.ostep, opath='.')
u, w, t = [np.asarray(f) for f in fields]

print('final Nusselt number:', float(solver.nusselt([u, w, t])))

# Plot final temperature (perturbation + conduction profile, Delta T = Lz)
T_full = t + (pm.Lz - z)[None, :]
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.pcolormesh(x[:, 0], z, T_full.T, cmap='RdBu_r', shading='gouraud')
ax.set_aspect('equal')
ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_title('Temperature')
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('temperature.png', dpi=200)
plt.close()
