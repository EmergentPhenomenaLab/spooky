"""Re-converge a SPECTER periodic orbit natively and compute its Floquet spectrum.

Default target: PO1 at Ra = 2e7 (``RB/ra2e7/512-512/1721_ra1.9e7``, SPECTER 512x512,
dt = 1e-3, Newton iteration 11: |F| = 3.9e-4, T = 10.8344, sx ~ 0).  The orbit is
converted with ``rb_convert`` to the native grid and polished with
``UPONewtonSolver`` (T and sx free, GMRES + hookstep, exact JVP tangents under JAX)
down to a relative residual ||Phi_T(U) - U||/||U|| < ``--tol``.  Then
``floquet_multipliers`` (Arnoldi on the exact tangent map) at the converged orbit,
printed next to SPECTER's ``floq/floq_n20.npy`` for the same orbit.

    NUMPY_BACKEND=jax python newton_po1.py --grid 256x97 --dt 2e-3 --out results/newton_po1_256x97
    ... --restart            # skip Newton, load <out>/converged.npz, run Floquet only

Free-fall units in the SPECTER box (Lz = pi), so T, dt and sx are SPECTER's.
``--norm quadrature`` (default) makes the Newton/GMRES inner products the physical
RMS norm (``rb_convert.quadrature_weights``); ``--norm plain`` uses the unweighted
collocation values.
"""
import argparse
import os
import re
import time

parser = argparse.ArgumentParser()
parser.add_argument('--ref', default='/share/data4/jcullen/RB/ra2e7/512-512/1721_ra1.9e7')
parser.add_argument('--iN', type=int, default=None, help='SPECTER Newton iteration to load (default: last)')
parser.add_argument('--ra', type=float, default=2e7)
parser.add_argument('--grid', default='256x97')
parser.add_argument('--dt', type=float, default=2e-3, help='timestep, free-fall units')
parser.add_argument('--norm', default='quadrature', choices=['quadrature', 'plain'],
                    help='inner product of the Newton state (DynSys weights)')
parser.add_argument('--tol', type=float, default=1e-9, help='Newton stop: ||F||/||U|| below this')
parser.add_argument('--tol-gmres', type=float, default=1e-3)
parser.add_argument('--n-gmres', type=int, default=300)
parser.add_argument('--max-newton', type=int, default=12)
parser.add_argument('--n-floquet', type=int, default=20, help='Arnoldi vectors (0 = skip)')
parser.add_argument('--floquet-tol', type=float, default=1e-10)
parser.add_argument('--out', default='results/newton_po1')
parser.add_argument('--restart', action='store_true')
parser.add_argument('--resume', action='store_true',
                    help='continue an interrupted Newton from <out>/newton_state.npz')
parser.add_argument('--threads', type=int, default=None, help='XLA CPU threads')
args = parser.parse_args()

if args.threads:
    os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
                               f' --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={args.threads}')
os.environ.setdefault('NUMPY_BACKEND', 'jax')

import numpy as np
from spooky._backend import JAX_ENABLED
if JAX_ENABLED:
    import jax
    jax.config.update('jax_enable_x64', True)
    print(f'jax {jax.__version__} {jax.devices()}', flush=True)
import spooky as sp
from spooky import rb_convert as rc
from spooky.solvers import RayleighBenard2D
from spooky.methods import UPONewtonSolver

Ra, Pr = args.ra, 1.0
Nx, Nz = (int(v) for v in args.grid.lower().split('x'))
out = args.out
os.makedirs(out, exist_ok=True)


def load_specter_newton(ref, iN=None):
    ''' (iN, |F|, T, sx, fields) of one Newton iteration of a pySPEC/SPECTER run. '''
    rows = np.loadtxt(os.path.join(ref, 'prints', 'solver.txt'), delimiter=',', skiprows=1, ndmin=2)
    row = rows[-1] if iN is None else rows[rows[:, 0] == iN][0]
    iN = int(row[0])
    pm = open(os.path.join(ref, 'params.py')).read()
    shape = tuple(int(re.search(rf'^{k}\s*=\s*(\d+)', pm, re.M).group(1)) for k in ('Nx', 'Nz'))
    fields = rc.load_specter_fields(os.path.join(ref, 'output', f'iN{iN:02}'), 0, shape)
    return iN, row[1], row[2], row[3], fields


def norm(fields):
    return np.sqrt(sum(float(grid.average(np.asarray(f) ** 2)) for f in fields))


grid = sp.Grid2D_wall(2 * np.pi, np.pi, Nx, Nz, args.dt)
solver = RayleighBenard2D(grid, ra=Ra, pr=Pr, bc='no-slip')
weights = rc.quadrature_weights(grid) if args.norm == 'quadrature' else None
ns = UPONewtonSolver(solver, T=1.0, sx=0.0, weights=weights,
                     N_newt=args.max_newton, N_gmres=args.n_gmres,
                     tol_gmres=args.tol_gmres, tol_improve=1.0, mu0=1e-3, mu_inc=2.0, c=0.5,
                     save_outputs='none', save_balance='none',
                     **{k: os.path.join(out, k) for k in
                        ('newton_dir', 'gmres_dir', 'hookstep_dir', 'trust_region_dir', 'apply_A_dir',
                         'output_dir', 'balance_dir', 'converged_dir')})

if not args.restart:
    iN_s, F_s, T_s, sx_s, fs = load_specter_newton(args.ref, args.iN)
    print(f'SPECTER {args.ref}\n  iN={iN_s} |F|={F_s:.3e} T={T_s:.12f} sx={sx_s:.3e} '
          f'Nu={rc.specter_nusselt(fs, Ra, Pr):.5f}', flush=True)
    fn = rc.specter_to_native(fs, grid)
    T, sx = T_s, rc.shift_to_native(sx_s, grid)
    U0 = ns.flatten(fn)
    ns.tol_newt = args.tol * np.linalg.norm(U0)
    print(f'native {Nx}x{Nz}, dt={args.dt:g} ({solver._nsteps(T)} steps/period), '
          f'Nu={float(solver.nusselt(fn)):.5f}, |U|={np.linalg.norm(U0):.4e} ({args.norm} norm), '
          f'tol_newt={ns.tol_newt:.3e}', flush=True)

    state_file = os.path.join(out, 'newton_state.npz')
    X = ns.form_X(U0, T, sx)
    if args.resume and os.path.exists(state_file):
        try:
            st = np.load(state_file)
            X, ns.restart_iN = st['X'], int(st['iN'])
            print(f'resuming after Newton iteration {ns.restart_iN} (|F| = {float(st["F"]):.3e})', flush=True)
        except Exception as e:                      # half-written checkpoint: start over
            print(f'could not load {state_file} ({type(e).__name__}); starting from the SPECTER orbit', flush=True)

    def checkpoint(iN, X, F):
        np.savez(state_file, X=X, iN=iN, F=F)
        print(f'  Newton iteration {iN}: |F| = {F:.3e}, |F|/|U| = {F/np.linalg.norm(X[:-2]):.3e}, '
              f'T = {X[-2]:.9f}, {(time.time()-t0)/60:.1f} min', flush=True)

    t0 = time.time()
    X = ns.run_newton(X, on_iteration=checkpoint)
    t_newton = time.time() - t0
    U, T, sx = ns.unpack_X(X)
    fields = ns.unflatten(U)
    UT = ns.translate(ns.evolve(U, T), sx)
    res = np.linalg.norm(U - UT) / np.linalg.norm(U)
    print(f'Newton done in {t_newton/60:.1f} min: ||F||/||U|| = {res:.3e} ({args.norm} norm)  '
          f'T = {T:.12f}  sx = {sx:.9e}  Nu = {float(solver.nusselt(fields)):.6f}', flush=True)
    np.savez(os.path.join(out, 'converged.npz'), u=fields[0], w=fields[1], th=fields[2],
             T=T, sx=sx, Ra=Ra, Nx=Nx, Nz=Nz, Lz=grid.Lz, dt=args.dt, norm=args.norm,
             residual=res, t_newton=t_newton)
else:
    d = np.load(os.path.join(out, 'converged.npz'))
    fields = [d['u'], d['w'], d['th']]
    T, sx = float(d['T']), float(d['sx'])
    print(f'loaded {out}/converged.npz: residual {float(d["residual"]):.3e}, T = {T:.12f}', flush=True)

if args.n_floquet > 0:
    t0 = time.time()
    eig, vec, Q = ns.floquet_multipliers(fields, T, args.n_floquet, args.floquet_tol, sx=sx)
    order = np.argsort(-np.abs(eig))
    eig = eig[order]
    np.savez(os.path.join(out, 'floquet.npz'), multipliers=eig, eigvec_H=vec[:, order], Q=Q,
             T=T, exponents=np.log(np.abs(eig)) / T)
    ref_file = os.path.join(args.ref, 'floq', f'floq_n{args.n_floquet}.npy')
    ref = np.load(ref_file) if os.path.exists(ref_file) else None
    if ref is not None:
        ref = ref[np.argsort(-np.abs(ref))]
    print(f'Floquet multipliers (Arnoldi, {len(eig)} vectors, {(time.time()-t0)/60:.1f} min); '
          f'exponent = ln|mu|/T with T = {T:.6f}', flush=True)
    print(f'{"native mu":>28s} {"|mu|":>10s} {"exp":>9s} | {"SPECTER mu":>28s} {"|mu|":>10s}')
    for i in range(min(len(eig), 12)):
        r = f'{ref[i].real:14.6f}{ref[i].imag:+13.6f}j {abs(ref[i]):10.6f}' if ref is not None and i < len(ref) else ''
        print(f'{eig[i].real:14.6f}{eig[i].imag:+13.6f}j {abs(eig[i]):10.6f} {np.log(abs(eig[i]))/T:9.5f} | {r}', flush=True)
