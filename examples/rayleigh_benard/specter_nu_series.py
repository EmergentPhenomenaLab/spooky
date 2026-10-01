"""Long-time Nu and KE of a SPECTER DNS run from all its field snapshots, with the same
flux definition (Nu = 1 + <w th>/(kappa dT/dz), SPECTER units) as the native sweep, so
the three codes can be compared over comparable averaging windows.

    python specter_nu_series.py /share/data4/jcullen/specter_runs/ra2e6/256-103/seed1 --tavg 300 [--stride 2]
"""
import argparse
import glob
import os
import re
import time

import numpy as np
from spooky import rb_convert as rc

parser = argparse.ArgumentParser()
parser.add_argument('run')
parser.add_argument('--tavg', type=float, default=300.0)
parser.add_argument('--stride', type=int, default=1, help='use every stride-th snapshot')
parser.add_argument('--out', default=None, help='npz to save the series to')
args = parser.parse_args()

pm = open(os.path.join(args.run, 'params.py')).read()
get = lambda k: float(re.search(rf'^{k}\s*=\s*([0-9.eE+-]+)', pm, re.M).group(1))
Nx, Nz, dt, ostep = int(get('Nx')), int(get('Nz')), get('dt'), int(get('ostep'))
# Ra from the viscosity the binary actually used (params.py can be stale): nu = pi^2 sqrt(Pr/Ra), Pr = 1
inp = open(os.path.join(args.run, 'parameter.inp')).read()
nu_s = float(re.search(r'^nu\s*=\s*([0-9.eE+-]+)', inp, re.M).group(1))
Ra = (np.pi ** 2 / nu_s) ** 2
idx = sorted(int(re.search(r'th\.(\d+)\.out$', f).group(1)) for f in glob.glob(os.path.join(args.run, 'output', 'th.*.out')))
idx = idx[::args.stride]
t = np.array(idx) * ostep * dt
print(f'{args.run}: {Nx}x{Nz} dt={dt:g} ostep={ostep} Ra={Ra:.4g} (from parameter.inp nu={nu_s:.6g}); {len(idx)} snapshots, t = {t[0]:.1f}..{t[-1]:.1f}', flush=True)

t0 = time.time()
nu = np.full(len(idx), np.nan)
for k, i in enumerate(idx):
    try:
        fs = rc.load_specter_fields(os.path.join(args.run, 'output'), i, (Nx, Nz))
        nu[k] = rc.specter_nusselt(fs, Ra, 1.0)
    except Exception as e:
        print(f'  snapshot {i}: {type(e).__name__}', flush=True)
    if (k + 1) % 1000 == 0:
        print(f'  {k+1}/{len(idx)} ({time.time()-t0:.0f} s)', flush=True)

bal = np.loadtxt(os.path.join(args.run, 'balance.txt'))
ke_ff = 0.5 * bal[:, 1] / np.pi ** 2          # SPECTER velocity unit -> free-fall (U_ff = pi)


def block_se(v, nb=10):
    means = np.array([b.mean() for b in np.array_split(v, nb)])
    return means.std(ddof=1) / np.sqrt(nb)


ok = np.isfinite(nu)
for t1 in (args.tavg, 1000.0):
    m = ok & (t > t1)
    mb = bal[:, 0] > t1
    if m.sum() > 20:
        print(f'  t > {t1:g}: Nu_flux = {nu[m].mean():.4f} +- {block_se(nu[m]):.4f} (std {nu[m].std():.3f}, {m.sum()} snapshots)'
              f'   KE_ff = {ke_ff[mb].mean():.5f}   last-200-snapshot mean = {nu[ok][-200:].mean():.4f}', flush=True)
if args.out:
    np.savez(args.out, t=t, nu_flux=nu, t_bal=bal[:, 0], ke_ff=ke_ff, Ra=Ra)
