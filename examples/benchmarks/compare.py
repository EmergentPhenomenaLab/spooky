"""Collect the benchmark outputs into one table and one figure.

Inputs
------
- SPECTER benchmark tables (GHOST ``bench=1`` format, time per step) in
  ``--specter`` (default /share/data4/jcullen/specter_runs/bench).
- ``bench-cpu-*.out`` / ``bench-gpu-*.out`` written by ``job_bench_cpu.sh`` and
  ``job_bench_gpu.sh`` (native RB solver, Dedalus, KolmogorovFlow).

    python compare.py results/bench-cpu-44506.out results/bench-gpu-44507.out
"""
import argparse
import glob
import os
import re

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('outs', nargs='+', help='job output files')
parser.add_argument('--specter', default='/share/data4/jcullen/specter_runs/bench')
parser.add_argument('--fig', default='results/rb_cost_per_step.png')
args = parser.parse_args()


def parse_specter(path):
    ''' {label: [(nprocs, ms/step)]} from bench_<Nx>_<Nz>.txt (TWTIME is per step). '''
    out = {}
    for f in sorted(glob.glob(os.path.join(path, 'bench_*.txt'))):
        d = np.loadtxt(f, skiprows=1, ndmin=2)
        label = os.path.basename(f)[len('bench_'):-len('.txt')].replace('_', 'x')
        out[label] = [(int(r[4]), 1e3 * r[8]) for r in d]
    return out


def parse_outs(files):
    ''' native[(device, threads, dtype)][grid] = dict(fwd, jvp, grad, evolve)
        dedalus[(scheme, grid)] = [(np, ms)];  kolmog[(backend, threads, dtype)][N] = dict '''
    native, dedalus, kolmog = {}, {}, {}
    for fn in files:
        section = None
        device, dtype = 'cpu', 'float64'
        for line in open(fn):
            m = re.match(r'=== (.*) ===', line)
            if m:
                head = m.group(1)
                threads = int(re.search(r'(\d+) threads', head).group(1)) if 'threads' in head else None
                if head.startswith('native RB'):
                    section = 'native'
                elif head.startswith('KolmogorovFlow'):
                    section = 'kolmog'
                elif head.startswith('Dedalus'):
                    section = 'dedalus'
                else:
                    section = None
                continue
            if line.startswith('jax '):
                device = 'gpu' if 'Cuda' in line else 'cpu'
                dtype = line.split('dtype')[-1].strip()
                continue
            if line.startswith('numpy '):
                device, dtype = 'numpy', 'float64'
                continue
            m = re.match(r'dedalus (\S+)\s+(\d+)x(\d+)\s+np=\s*(\d+) \| build.*\|\s+([\d.]+) ms/step', line)
            if m:
                dedalus.setdefault((m.group(1), f'{m.group(2)}x{m.group(3)}'), []).append(
                    (int(m.group(4)), float(m.group(5))))
                continue
            m = re.match(r'\s*(\d+)x(\d+)\s+\|(.*)', line)
            if m and section in ('native', 'kolmog'):
                vals = {k: float(v) for k, v in re.findall(r'(fwd|jvp|grad|evolve)\s+([\d.]+|nan)', m.group(3))}
                grid = f'{m.group(1)}x{m.group(2)}'
                tgt = native if section == 'native' else kolmog
                tgt.setdefault((device, threads, dtype), {})[grid] = vals
    return native, dedalus, kolmog


specter = parse_specter(args.specter)
native, dedalus, kolmog = parse_outs(args.outs)

print('## SPECTER (ms/step, best over nprocs; np=1 or smallest measured)')
for label, rows in specter.items():
    best = min(rows, key=lambda r: r[1]); first = rows[0]
    print(f'  {label:>8s}: np={first[0]:2d} {first[1]:7.2f} | best np={best[0]:2d} {best[1]:6.2f}')

print('\n## native RB solver (ms/step)')
for (device, threads, dtype), grids in native.items():
    for grid, v in grids.items():
        cols = '  '.join(f'{k} {v[k]:7.2f}' for k in ('fwd', 'jvp', 'grad', 'evolve') if k in v)
        print(f'  {device:5s} threads={str(threads):>4s} {dtype:8s} {grid:>8s}: {cols}')

print('\n## Dedalus (ms/step)')
for (scheme, grid), rows in dedalus.items():
    print(f'  {scheme:6s} {grid:>8s}: ' + '  '.join(f'np={n}:{t:.2f}' for n, t in rows))

print('\n## KolmogorovFlow (ms/step)')
for (device, threads, dtype), grids in kolmog.items():
    for grid, v in grids.items():
        cols = '  '.join(f'{k} {v[k]:8.3f}' for k in ('fwd', 'jvp', 'grad', 'evolve') if k in v)
        print(f'  {device:5s} threads={str(threads):>4s} {dtype:8s} {grid:>10s}: {cols}')

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'
C = {'specter': '#2a78d6', 'dedalus': '#eb6834', 'native': '#1baf7a', 'gpu': '#4a3aa7'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False})

panels = [('256-class grids', ['256x231', '256x103'], '256x129'),
          ('512-class grids', ['512x512'], '512x257')]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6), sharey=True, facecolor=SURFACE)
for ax, (title, sp_labels, grid) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    for label, ls, mk in zip(sp_labels, ['-', '--'], ['o', 's']):
        if label in specter:
            n, t = zip(*specter[label])
            ax.plot(n, t, ls, marker=mk, ms=3.5, lw=1.6, color=C['specter'], label=f'SPECTER {label} (RK2, MPI)')
    for scheme, ls, mk in [('SBDF2', '-', 'o'), ('RK222', '--', 's')]:
        if (scheme, grid) in dedalus:
            n, t = zip(*dedalus[(scheme, grid)])
            ax.plot(n, t, ls, marker=mk, ms=3.5, lw=1.6, color=C['dedalus'], label=f'Dedalus {grid} ({scheme}, MPI)')
    rows = sorted((th, g[grid]) for (dev, th, dt), g in native.items()
                  if dev == 'cpu' and th is not None and grid in g)
    if rows:
        th = [r[0] for r in rows]
        ax.plot(th, [r[1]['fwd'] for r in rows], '-', marker='o', ms=3.5, lw=1.6, color=C['native'],
                label=f'native {grid} forward (JAX CPU threads)')
        ax.plot(th, [r[1]['jvp'] for r in rows], '--', marker='s', ms=3.5, lw=1.6, color=C['native'],
                label=f'native {grid} JVP (JAX CPU threads)')
    for (dev, th, dt), g in native.items():
        if dev == 'gpu' and dt == 'float64' and grid in g:
            ax.axhline(g[grid]['fwd'], color=C['gpu'], lw=1.6, ls='-')
            ax.axhline(g[grid]['jvp'], color=C['gpu'], lw=1.6, ls='--')
            ax.text(1, 1.06 * g[grid]['fwd'], f"L4 forward {g[grid]['fwd']:.2f} ms", color=INK, va='bottom', fontsize=8)
            ax.text(1, 1.06 * g[grid]['jvp'], f"L4 JVP {g[grid]['jvp']:.2f} ms", color=INK, va='bottom', fontsize=8)
            ax.plot([], [], '-', color=C['gpu'], lw=1.6, label=f'native {grid}, one L4 GPU (float64)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(['1', '2', '4', '8', '16', '32'])
    ax.set_xlim(0.9, 55)
    ax.grid(True, which='major', color=GRID, lw=0.6)
    ax.set_xlabel('CPU cores (MPI ranks or threads)')
    ax.set_title(title, loc='left', fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
axes[0].set_ylabel('wall time per time step (ms)')
fig.suptitle('2D Rayleigh-Benard: cost per time step on c3 (2x Xeon 6230) and one L4', x=0.01, ha='left',
             fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(os.path.dirname(args.fig) or '.', exist_ok=True)
fig.savefig(args.fig, dpi=200, facecolor=SURFACE)
print(f'\nfigure: {args.fig}')
