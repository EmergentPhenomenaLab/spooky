# Benchmarks

Cost per time step of the native solvers, under JAX on CPU/GPU and under NumPy, compared with
SPECTER and Dedalus.

## Scripts

- **`bench_kolmogorov.py`**: per-step cost of `KolmogorovFlow`.
  - Under NumPy: `solver.evolve`.
  - Under JAX: the jitted `ad_evolve` forward map, one JVP and one gradient through it, and
    `solver.evolve`.
  - `--dtype float32` for the single-precision comparison.
- **`bench_rb.py`**: the same for `RayleighBenard2D`.
  - Problem: box 2×1, no-slip, Pr = 1, `--ra` (default 1e6), dt = 1e-3 free-fall units.
  - `--evolve` also times `solver.evolve`; `--mem` reports compiled memory.
  - With `NUMPY_BACKEND=numpy` it times `solver.evolve` only.
- **`bench_dedalus_rb.py`**: Dedalus v3 on the same RB problem, fixed `dt`, no analysis output.
  Run it inside the `dedalus3` micromamba environment with `mpiexec`.
- **`job_bench_cpu.sh`**: one exclusive CPU node. The default is `c3`, where the SPECTER tables of
  `specter_runs/bench` were measured. It runs native thread scaling, NumPy, Kolmogorov, and Dedalus
  MPI scaling.
- **`job_bench_gpu.sh`**: one GPU (default an L4 on `c6`; `sbatch -w c4` for an A100). It runs
  native RB, Kolmogorov, and `examples/rayleigh_benard/specter_orbit_dt_study.py`.
- **`compare.py`**: merges the job outputs with the SPECTER tables into one table and one figure.

**Environment.** Override it with `SPOOKY_ENV=... sbatch job_bench_gpu.sh`.

**Frozen code.** Both environments import the live worktree `spooky_AD/spooky`. To benchmark a
fixed commit, set `PYTHONPATH` (and `SPOOKY_DIR`) to an extracted copy:
`git archive <commit> spooky examples | tar -x -C <dir>`.

**Old outputs.** The raw outputs of the 2026-09-21 runs (diffusive-units solver, before the
real-valued operators) are kept in the local `AD-diffusive` branch under
`examples/benchmarks/results/`.

## Reading the output

**`bench_rb.py` / `bench_kolmogorov.py`, JAX.** A header line `jax <version> devices [...] dtype
...`, then one line per grid:

```
 NxxNz   | build <s> | ops <GB> | fwd <ms> ms/step | jvp <ms> | grad <ms> | evolve <ms>
           temp: fwd <MB> | jvp <MB> | grad <MB>  -> tape ~<MB> MB/step
```

| field | meaning |
|---|---|
| `build` | RB only: building the implicit operators (NumPy, once per `(dt, Ra, Pr)`) |
| `ops` | RB only: memory of the implicit operators (fixed ones plus one SBDF2 set) |
| `fwd` | jitted `ad_evolve` (`lax.scan`), per step |
| `jvp` | one forward-mode tangent through it, per step. This is the cost of one GMRES or Arnoldi iteration divided by the steps per period. |
| `grad` | one reverse-mode gradient, per step (`nan` with `--no-grad`) |
| `evolve` | `solver.evolve`, the per-step Python loop that `DynSys` and Newton use |
| `temp ... tape` | (`--mem`) compiled temporary memory per program. The gradient's memory divided by the number of steps is the reverse-mode memory per step. |

**Timing method.** Each number is the minimum over `--repeat` runs after one compile run.

**NumPy.** One line per grid: `NxxNz | evolve <ms> ms/step`.

**`bench_dedalus_rb.py`.** One line from rank 0:
`dedalus <scheme> <grid> np=<ranks> | build <s> | <ms> ms/step`.

**`compare.py <job outputs>`.** Prints these tables and writes `results/rb_cost_per_step.png`:
- SPECTER ms/step (first and best process count);
- native RB per device, thread count and dtype;
- Dedalus per scheme and grid;
- Kolmogorov.

## Reference numbers

**Native RB, JVP per step** (2026-10-01, real-valued operators, Ra = 2e7, SPECTER box;
`/share/data4/jcullen/rb_bench/3b`):

| | 256×97 | 512×193 | 512×257 |
|---|---|---|---|
| A100 | 0.43 ms | 1.16 ms | 1.65 ms |
| L4 | 1.11 ms | 7.31 ms | 12.2 ms |
| JAX CPU (c3, 10 cores) | 25 ms | 142 ms | — |

**Older comparison** (2026-09-21, diffusive solver, before the real-valued operators), in ms/step:

| code | 256-class grid | 512-class grid |
|---|---|---|
| SPECTER (best on c3) | 3.4 (256×103), 5.5 (256×231, 15 ranks) | 13.4 (512×512, 30 ranks) |
| Dedalus SBDF2 | 4.2 (256×129, 40 ranks) | 12.1 (512×257) |

Native JAX on CPU barely scales with threads at the 256 class (~16 ms/step at any thread count).

**Implications.**
- One Newton iteration for PO1 takes 2–3 h with SPECTER, and ~5–10 min natively on an A100 or L4
  at 256×97.
- At 512×257 the L4 only breaks even with SPECTER.
- dt is limited by the advective CFL condition in x (`specter_orbit_dt_study.py`), so the implicit
  scheme does not allow a larger dt than SPECTER at equal Nx.
