"""Table + figure for the DNS Nu(Ra) sweep of ``nu_sweep.py`` against the SPECTER
reference CSV and the Dedalus runs.

    python nu_sweep_compare.py results/nu_sweep_256x97 [--tavg 300] [--fig nu_vs_ra.png]
"""
import argparse
import glob
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('dirs', nargs='+', help='output dirs of nu_sweep.py (one per grid)')
parser.add_argument('--tavg', type=float, default=300.0)
parser.add_argument('--ref-csv', default='/share/data4/jcullen/specter_runs/Imgs/nu_comparison/nu_dns_gradient.csv')
parser.add_argument('--fig', default=None)
args = parser.parse_args()

# Dedalus (dedalus/rbc2D, RK222 + CFL, Nu = 1 + sqrt(Ra) <w b> corrected from the script's diagnostic,
# t > 300 of 1000): grid -> (Ra, Nu, block SE)
DEDALUS = {'256x103': (2e6, 9.0542, 0.0236), '128x64 (2e6)': (2e6, 9.0515, 0.0235), '128x64 (9e5)': (9e5, 7.4941, 0.0046)}

parser_specter = os.path.join(os.path.dirname(args.dirs[0].rstrip('/')), 'specter_series')
ref = np.genfromtxt(args.ref_csv, delimiter=',', skip_header=1, usecols=(0, 1, 2))

# long-time flux Nu of the SPECTER DNS runs themselves (specter_nu_series.py): {Ra: [(label, mean, SE, KE)]}
flux_ref = {}
for f in sorted(glob.glob(os.path.join(parser_specter, '*.npz'))):
    z = np.load(f)
    t, nu = z['t'], z['nu_flux']
    m = np.isfinite(nu) & (t > args.tavg)
    if m.sum() < 10:
        continue
    means = np.array([b.mean() for b in np.array_split(nu[m], 10)])
    ke = float(z['ke_ff'][z['t_bal'] > args.tavg].mean()) if len(z['t_bal']) else np.nan
    flux_ref.setdefault(round(float(z['Ra']), -2), []).append(
        (os.path.basename(f)[:-4], nu[m].mean(), means.std(ddof=1) / np.sqrt(10), ke, t[m][-1]))


def block_se(v, nb=10):
    means = np.array([b.mean() for b in np.array_split(v, nb)])
    return means.std(ddof=1) / np.sqrt(nb) if len(v) >= nb else np.nan


rows = []
for d in args.dirs:
    for f in sorted(glob.glob(os.path.join(d, 'ra*.npz')), key=lambda f: float(np.load(f)['Ra'])):
        z = np.load(f)
        Ra, t = float(z['Ra']), z['t']
        m = t > args.tavg
        if m.sum() < 10:
            continue
        r = ref[np.isclose(ref[:, 0], Ra, rtol=1e-6)]
        rows.append(dict(grid=f'{int(z["Nx"])}x{int(z["Nz"])}', Ra=Ra, tend=t[-1], dt=float(z['dt_ff']),
                         nu_flux=z['nu_flux'][m].mean(), se_flux=block_se(z['nu_flux'][m]), std=z['nu_flux'][m].std(),
                         nu_grad=z['nu_grad'][m].mean(), nu_wall=z['nu_wall'][m].mean(), ke=z['ke'][m].mean(),
                         umax=z['umax'][m].max(), ref=r[0, 1] if len(r) else np.nan, ref_std=r[0, 2] if len(r) else np.nan))

print(f'{"grid":>8s} {"Ra":>8s} {"t_end":>6s} {"dt":>7s} | {"Nu_flux":>8s} {"+-SE":>7s} {"std":>6s} {"Nu_grad":>8s} {"Nu_wall":>8s} '
      f'{"KE_ff":>8s} {"umax":>6s} | {"CSV":>8s} {"diff%":>6s} | {"SPECTER flux (run, Nu +- SE, KE, t_end)":s}')
for r in rows:
    diff = 100 * (r['nu_flux'] / r['ref'] - 1) if np.isfinite(r['ref']) else np.nan
    fr = flux_ref.get(round(r['Ra'], -2), [])
    frs = '; '.join(f'{lab.replace("_256-103", "").replace("_512-512", "/512")}: {m:.4f}+-{se:.4f} (KE {ke:.4f}, t {te:.0f}) '
                    f'[{100*(r["nu_flux"]/m-1):+.1f}%]' for lab, m, se, ke, te in fr)
    print(f'{r["grid"]:>8s} {r["Ra"]:8.2e} {r["tend"]:6.0f} {r["dt"]:7.1e} | {r["nu_flux"]:8.4f} {r["se_flux"]:7.4f} {r["std"]:6.3f} '
          f'{r["nu_grad"]:8.4f} {r["nu_wall"]:8.4f} {r["ke"]:8.5f} {r["umax"]:6.3f} | {r["ref"]:8.4f} {diff:6.1f} | {frs}')
print('\nDedalus: ' + '; '.join(f'{k}: Ra={v[0]:.0e} Nu={v[1]:.4f}+-{v[2]:.4f}' for k, v in DEDALUS.items()))

if args.fig:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    INK, MUTED, GRID, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'
    C = {'specter': '#2a78d6', 'dedalus': '#eb6834', 'native': '#1baf7a'}
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), facecolor=SURFACE)
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=0.6)
    a1.errorbar(ref[:, 0], ref[:, 1], yerr=ref[:, 2], fmt='o', ms=3, lw=1, capsize=2, color=C['specter'],
                label='SPECTER DNS (paper), mean +- std of the series')
    for g in sorted({r['grid'] for r in rows}):
        rr = [r for r in rows if r['grid'] == g]
        a1.errorbar([r['Ra'] for r in rr], [r['nu_flux'] for r in rr], yerr=[r['std'] for r in rr], fmt='s', ms=4, lw=1,
                    capsize=2, color=C['native'], label=f'native {g} (SBDF2), mean +- std')
    a1.errorbar([v[0] for v in DEDALUS.values()], [v[1] for v in DEDALUS.values()], fmt='^', ms=5, color=C['dedalus'],
                label='Dedalus (rbc2D runs)')
    if flux_ref:
        xs = [Ra for Ra, v in flux_ref.items() for _ in v]; ys = [m for v in flux_ref.values() for _, m, *_ in v]
        a1.plot(xs, ys, 'D', ms=4, mfc='none', mew=1.2, color=C['specter'], label='SPECTER runs, long-time flux Nu')
    a1.set_xscale('log'); a1.set_yscale('log'); a1.set_xlabel('Ra'); a1.set_ylabel('Nu')
    a1.set_title('Nu(Ra), DNS from noise', loc='left', fontsize=10)
    a1.legend(frameon=False, fontsize=7.5, loc='upper left')
    for r in rows:
        d = 100 * (r['nu_flux'] / r['ref'] - 1)
        a2.errorbar(r['Ra'], d, yerr=100 * np.hypot(r['se_flux'], r['ref_std'] / np.sqrt(200)) / r['ref'], fmt='s', ms=4,
                    color=C['native'], capsize=2)
    for k, v in DEDALUS.items():
        rr = ref[np.isclose(ref[:, 0], v[0], rtol=1e-6)]
        if len(rr):
            a2.plot(v[0], 100 * (v[1] / rr[0, 1] - 1), '^', ms=5, color=C['dedalus'])
    for Ra, v in flux_ref.items():
        rr = ref[np.isclose(ref[:, 0], Ra, rtol=1e-6)]
        for _, m, *_ in v:
            if len(rr):
                a2.plot(Ra, 100 * (m / rr[0, 1] - 1), 'D', ms=4, mfc='none', mew=1.2, color=C['specter'])
    a2.axhline(0, color=C['specter'], lw=1.2)
    a2.plot([], [], 'D', ms=4, mfc='none', mew=1.2, color=C['specter'], label='SPECTER long-time flux Nu vs its CSV')
    a2.plot([], [], 's', ms=4, color=C['native'], label='native 256x97'); a2.plot([], [], '^', ms=5, color=C['dedalus'], label='Dedalus')
    a2.legend(frameon=False, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3)
    a2.set_ylabel('Nu / Nu_CSV - 1  (%)')
    a2.set_xscale('log'); a2.set_xlabel('Ra')
    a2.set_title("Difference to the paper's DNS CSV (np.gradient estimator)", loc='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=200, facecolor=SURFACE)
    print('figure:', args.fig)
