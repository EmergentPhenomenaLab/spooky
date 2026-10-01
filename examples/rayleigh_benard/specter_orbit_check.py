"""Convert a converged SPECTER/BOUSS periodic orbit to the native solver and
integrate one period: recurrence residual ||Phi_T(U)-U||/||U||.

Both codes use free-fall units, so T, dt and sx carry over unchanged in the
SPECTER box (Lz = pi); pass another Lz to test the length rescaling
(``python specter_orbit_check.py 1``).

Result on 2026-10-01 for PO1 at Ra = 1.9977e7 (SPECTER 512x512, T = 10.8313)
converted to 256x129, dt = 1e-3: residual 5.675e-5 (u 3.1e-5, w 4.4e-5,
theta 7.8e-5), identical for Lz = pi and Lz = 1; Nu 10.5598 -> 10.5599, KE ratio
1.00000; half-period distance 0.31; ~4 min on c3 with the NumPy backend (4
threads).  Edit the paths/parameters below for another orbit.
"""
import os, sys, time
os.environ.setdefault('NUMPY_BACKEND', 'numpy')
import numpy as np
import spooky as sp
from spooky import rb_convert as rc
from spooky.solvers import RayleighBenard2D

Lz = float(sys.argv[1]) if len(sys.argv) > 1 else np.pi
d = '/share/data4/jcullen/RB/arclength/PO1_up/converged/'
Ra, Pr = 19977416.43527502, 1.0
T, sx_s, dt = 10.831331173674736, 5.47063270e-09, 1e-3
fs = rc.load_specter_fields(d, 2, (512, 512))
print('SPECTER Nu (volume, snapshot):', rc.specter_nusselt(fs, Ra, Pr))

Nx, Nz = 256, 129
grid = sp.Grid2D_wall(2 * Lz, Lz, Nx, Nz, dt)
solver = RayleighBenard2D(grid, ra=Ra, pr=Pr, bc='no-slip')
fn = rc.specter_to_native(fs, grid)
sx = rc.shift_to_native(sx_s, grid)
print(f'native grid {Nx}x{Nz}, Lz={Lz:.6f}, dt={dt}, T={T} ({solver._nsteps(T)} steps), sx={sx:.3e}')
print('native Nu of converted snapshot:', float(solver.nusselt(fn)))
ke0 = 0.5 * float(grid.average(fn[0]**2 + fn[1]**2))
norm0 = np.sqrt(sum(float(grid.average(f**2)) for f in fn))

t0 = time.time()
out = grid.translate(solver.evolve([f.copy() for f in fn], T=T, write_outputs=False), sx)
res = np.sqrt(sum(float(grid.average((a-b)**2)) for a, b in zip(out, fn))) / norm0
print(f'full period: Nu={float(solver.nusselt(out)):.4f}  KE ratio={0.5*float(grid.average(out[0]**2+out[1]**2))/ke0:.5f}  '
      f'recurrence residual ||Phi_T(U)-U||/||U|| = {res:.4e}  ({time.time()-t0:.0f}s)', flush=True)
# residual vs a half period for scale
half = solver.evolve([f.copy() for f in fn], T=0.5*T, write_outputs=False)
res_half = np.sqrt(sum(float(grid.average((a-b)**2)) for a, b in zip(half, fn))) / norm0
print(f'half period distance (scale reference): {res_half:.4e}')
np.savez('orbit_native.npz', u=fn[0], w=fn[1], th=fn[2], uT=out[0], wT=out[1], thT=out[2], T=T, sx=sx, Ra=Ra, Lz=Lz)
