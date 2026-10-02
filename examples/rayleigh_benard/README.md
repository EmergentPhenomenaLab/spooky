# Rayleigh–Bénard examples (native solver)

Scripts for the native 2D Rayleigh–Bénard solver `RayleighBenard2D` (Fourier in x, Chebyshev in z,
SBDF2 implicit–explicit time stepping) and for comparing it with SPECTER. This file explains what
each script computes, where its output goes, how to read that output, and which numbers are known
to be right.

## Conventions shared by all scripts

- **Free-fall units.** Time is the free-fall time, and the base temperature gradient is 1, so the
  temperature difference across the layer equals the layer height `Lz`. Viscosity and diffusivity
  are `nu = Lz^2 sqrt(Pr/Ra)` and `kappa = Lz^2 / sqrt(Pr Ra)` (`solver.nu`, `solver.kappa`).
- **The SPECTER box.** Most scripts use `Lx = 2 pi`, `Lz = pi`. In that box the native solver and
  SPECTER share units: periods `T`, time steps `dt` and shifts `sx` mean the same thing in both
  codes. With another `Lz`, lengths, velocities and `theta` scale by `Lz/pi` and times stay the same
  (`spooky.rb_convert`).
- **Fields** are `[u, w, theta]`, physical arrays of shape `(Nx, Nz)`. `theta` is the deviation
  from the conduction profile `T_cond = Lz - z`, so the full temperature is `theta + Lz - z`. Row
  `z = 0` (index 0) is the hot bottom wall, and all three fields vanish at both walls (no-slip).
- **Nusselt number.** `solver.nusselt(fields) = 1 + <w theta> / kappa` (volume average, flux
  definition). `nu_sweep.py` also reports the gradient and wall estimates, which agree with it for
  long runs.
- **Norms.** A relative residual `||Phi_T(U) - U|| / ||U||` depends on how u, w and theta are
  weighted (example below in `specter_orbit_check.py`). Quote residuals together with their norm.

## Running

```bash
module load python/3.11.3
export PYTHONPATH=/share/data4/jcullen/spooky_AD       # the editable install points at the main checkout
NUMPY_BACKEND=jax python newton_po1.py ...             # scripts enable float64 themselves
```

- **GPU:** `source /share/data4/jcullen/envs/spooky_jax_gpu/bin/activate` after the module load
  (L4s on `c6`, A100s on `c4`).
- **Plots:** matplotlib is broken under `python/3.11.3`. Use
  `source /share/data4/jcullen/envs/specter_env/bin/activate` for `time_marching.py`'s plot.
- **Jobs import the live worktree.** The `job_*.sh` scripts `cd` into this folder but do not set
  `PYTHONPATH`, and both environments import the live worktree `spooky_AD/spooky`. A queued job
  therefore runs whatever branch is checked out when it starts. For long runs, copy the code first
  (`git archive <commit> spooky | tar -x -C <dir>`) and set `PYTHONPATH=<dir>`.
- **Outputs.** Old outputs of the diffusive-units solver are in `results/`, which git ignores.
  Their files use different units and keys, so don't mix them with new runs.

## Scripts

### `time_marching.py` (+ `params.py`): one simulation from rest

Evolves a small random temperature perturbation at `Ra = 1e5` in the SPECTER box (128×65,
`dt = 0.02`, `T = 200`). Runs in about a minute with NumPy.

| output (in the working directory) | content |
|---|---|
| `balance.dat` | one row every `bstep` steps: time, kinetic energy `0.5 <u^2 + w^2>`, Nu |
| `uu.NNNN.npy`, `ww.NNNN.npy`, `th.NNNN.npy` | fields every `ostep` steps (`NNNN` = step number) |
| `temperature.png` | final full temperature `theta + Lz - z` |
| stdout | final Nusselt number |

**Expected result.** Convection starts at `t ≈ 10`. After a few overshoots the flow settles into
**two steady counter-rotating rolls** by `t ≈ 100`, with **Nu = 4.9943** and **KE = 0.3387**. The
old diffusive-units run gave the same Nu, and its KE `3431.6` converts to
`3431.6 / Ra * pi^2 = 0.3387`.

### `specter_orbit_check.py`: does the native solver hold a SPECTER orbit?

Loads the converged SPECTER PO1 (`Ra = 1.9977e7`, `T = 10.8313`, 512×512), interpolates it to
256×129, integrates exactly one period with `dt = 1e-3`, and measures how far the state is from
where it started. An optional argument sets `Lz`: `python specter_orbit_check.py 1` tests the
length rescaling. About 4 minutes with NumPy on `c3`.

| output | content |
|---|---|
| stdout | SPECTER Nu; native Nu of the converted state; after one period: Nu, KE ratio, recurrence residual; distance after half a period (for scale) |
| `orbit_native.npz` | `u, w, th` (start), `uT, wT, thT` (after one period), `T, sx, Ra, Lz` |

**Expected result (2026-10-01).**
- **Residual:** `5.675e-5`, per field `u 3.1e-5, w 4.4e-5, theta 7.8e-5`. That's ~5000× below the
  half-period distance (0.31).
- **Nu and KE:** Nu 10.5598 → 10.5599, KE ratio 1.00000.
- **Lz = pi and Lz = 1** give identical numbers.

What's left is interpolation error plus the difference between the two codes' discretisations;
Newton (`newton_po1.py`) removes it.

The archived diffusive-units solver reaches the same state to 1e-8 after the period, but reports
`3.83e-5`. In diffusive units velocities are ~1400× larger than `theta`, so its norm is almost pure
velocity: same orbit, different weighting. (An older archive note says `8.1e-5`; that came from an
earlier code version.)

### `specter_orbit_dt_study.py`: usable time step and resolution (JAX)

Same orbit, for every combination of `--grids` (default `256x97,256x129,256x193`) and `--dts`
(default `5e-4 ... 1.6e-2`). For each, `dt` is adjusted so a whole number of steps makes up exactly
one period. One table row per run:

| column | meaning |
|---|---|
| `dt_ff`, `steps` | the time step actually used, and the steps per period |
| `residual` | recurrence residual as above (free-fall norm) |
| `Nu(T)`, `KE ratio` | Nu and KE after one period (KE relative to the start) |
| `build s`, `run s`, `ms/step` | operator build time; run time **including compilation**; per step |

**How to read it.**
- At small `dt` the residual levels off at the interpolation plus code-difference error.
- Above some `dt` it grows like `dt^2` (the scheme is second order).
- `NaN` means the explicit advection went unstable: the time step is limited by the advective CFL
  condition in x, not by diffusion.

**Earlier results** (diffusive solver, 2026-09-21; the residuals are in the diffusive norm, ~1.5×
smaller than free-fall ones):
- `Nz = 97` is enough: the residual is the same to 4 digits for `Nz = 97/129/193`.
- `Nx = 256` gives ~`3.8e-5`; `Nx >= 384` gives `1.8e-5`; `Nx = 128` is under-resolved.
- Stable up to `dt ≈ 3e-3` at `Nx = 256` and `< 2e-3` at `Nx = 512`. So implicit time stepping
  gives **no** `dt` advantage over SPECTER at equal `Nx`.

**Free-fall results** (2026-10-01, L4, job 45106, PO1 at Ra = 1.9977e7, T = 10.8313;
`/share/data4/jcullen/rb_bench/item34/bench-gpu-ff-45106.out`):

| dt | residual (all three Nz) | Nu(T) |
|---|---|---|
| 5e-4 | 5.529e-5 | 10.5595 |
| 1e-3 | 5.675e-5 | 10.5599 |
| 2e-3 | 1.165e-4 | 10.5616 |
| 4e-3, 8e-3, 1.6e-2 | NaN (unstable) | |

- `Nz = 97`, `129` and `193` give the same residual and Nu to all printed digits at every `dt`.
- The floor (~5.5e-5) is the interpolation plus code-difference error; `dt = 2e-3` doubles it, and the
  stability limit at `Nx = 256` lies between 2e-3 and 4e-3. Production `dt = 2e-3` is therefore within
  a factor < 2 of the limit at Ra ≈ 2e7.

**Converged-orbit resolution check** (2026-10-01, PO4 at Ra = 1.4e7, T ≈ 1.623; native Newton to
3e-11 and Floquet n = 40 on each grid; `/share/data4/jcullen/RB_native/ra1.4e7/*/po4_rescheck_*`):

| grid, dt | T | Nu | leading pair \|mu\| |
|---|---|---|---|
| 256×97, 2e-3 | 1.622914574 | 15.881123 | 1.076549 |
| 256×97, 1.5e-3 | 1.622925411 | 15.880942 | 1.076560 |
| 384×129, 1.5e-3 | 1.622925032 | 15.880939 | 1.076559 |

- Grid 256×97 → 384×129 at equal dt: T changes by 2e-7 (relative), Nu by 2e-7, the multipliers
  by ~1e-6.
- dt 1.5e-3 → 2e-3: T changes by 7e-6, Nu by 1e-5, |mu| by 1e-5. The time step, not the grid,
  dominates the error, and both are far below what matters for identifying orbits.
- SPECTER (512², dt 1e-3) has T = 1.622939, 8e-6 away.

### `newton_po1.py` (+ `job_newton_po1.sh`, `job_newton_po1_ra.sh`): converge a periodic orbit natively, then Floquet

Starts from one Newton iteration of a SPECTER run (`--ref <run dir>`, `--iN`, default the last),
interpolates it to the native grid, and runs `UPONewtonSolver`: GMRES plus hookstep, with `T` and
`sx` free and exact JVP tangents under JAX. Newton stops once the relative residual is below
`--tol` (default `1e-9`). Then it computes `--n-floquet` Floquet multipliers with Arnoldi on the
exact tangent map.
- `--norm quadrature` (default) makes Newton's inner products the physical root-mean-square norm.
  `--norm plain` uses the raw Chebyshev grid values, which over-count the points near the walls.
- `--resume` restarts from the checkpoint; `--restart` skips Newton and recomputes Floquet from
  `converged.npz`.

Output directory `--out`:

| file | content |
|---|---|
| stdout | SPECTER start (`iN`, `|F|`, `T`, `sx`, Nu); native start; one line per accepted Newton step (`|F|`, `|F|/|U|`, `T`, elapsed time); final residual, `T`, `sx`, Nu; Floquet table |
| `newton_dir/newton.txt` | `iN, |F|, T, sx, |U|` per Newton step (`iN = 00` is the starting state; `|F|`, `|U|` in the chosen norm) |
| `gmres_dir/gmres_iNkk.txt` | `iG, error`: GMRES relative residual per Krylov step (stops at `--tol-gmres`, default `1e-3`) |
| `hookstep_dir/hookstep_iNkk.txt` | `iH, |F|, |F(x)+cAdx|` per trial step: the actual new residual vs the linear model's prediction. The step is accepted when the actual is at most the prediction. |
| `trust_region_dir/trust_region_iNkk.txt` | `Delta, mu, |y|, cond(A)`: trust radius, Levenberg parameter, step size, condition number of the GMRES matrix |
| `apply_A_dir/apply_A_iNkk.txt` | norms of the Newton matrix's blocks per GMRES step: `|U|, |dU|, |J dU|, |dU/dt|, |dUT/dT|, |dU/ds|, |dUT/ds|`, and the two constraint projections |
| `newton_state.npz` | checkpoint after every accepted step: `X = (U, T, sx)`, `iN`, `F` |
| `converged.npz` | `u, w, th`, `T`, `sx`, `Ra`, `Nx`, `Nz`, `Lz`, `dt`, `norm`, `residual`, `t_newton` |
| `floquet.npz` | `multipliers` (sorted by modulus), `eigvec_H`, `Q` (Arnoldi basis), `T`, `exponents = ln|mu| / T` |

**How to read it.**
- **Healthy Newton** converges quadratically: the number of correct digits roughly doubles each
  iteration.
- **A stall** shows in the hookstep log: the full step's actual `|F|` is far above its prediction,
  and the step shrinks until it does nothing.
- **Floquet multipliers:** `|mu| > 1` is an unstable direction. Two multipliers near 1 are neutral:
  the shift along the orbit and the shift in x.
- **Floquet table vs SPECTER:** the right-hand columns come from `<ref>/floq/floq_n<n>.npy`, only
  when that file exists for the same `n`.

**Known results.**
- **PO1 has a fold (saddle-node) near `Ra ≈ 2e7`.** Fixed-Ra Newton there stalls at a relative
  residual `3.1e-6` (diagnosed 2026-09-23; SPECTER's own Newton also stalls at 2e7 and 1.95e7). Use
  `Ra <= 1.9e7`, or arclength continuation across the fold.
- **Away from the fold** (archived diffusive solver with quadrature weights, 256×97, `dt = 2e-3`,
  A100):
  - `Ra = 1.9e7`: `3.7e-6 → 3.4e-9 → 4.4e-11` in 3 iterations (11.6 min).
  - `Ra = 1.7e7`: same behaviour, final `3e-11`.
- **Floquet at 1.9e7** (n = 20): `24.53`, `11.30` (complex pair), `4.20`, ...
- **Floquet at 2e7:** matches SPECTER to ~1e-4 for the leading multiplier (22.136 vs 22.134).
- **Use `n >= 40`.** With `n = 20` the weaker multipliers aren't converged: two 1.7e7 runs agree
  except for the 4th (5.01 vs 5.63).
- **Validation 3** (this solver, Ra = 1.9e7, 256×97, dt = 2e-3, Floquet `n = 40`, 2026-10-01;
  `/share/data4/jcullen/rb_bench/val3/`): from the SPECTER orbit (iN 16) both norms converge in
  **3 iterations** (~14.5 min on an L4) to the same orbit, T = 10.7808887.
  - Quadrature: 3.8e-6 → 4.4e-9 → 1.9e-11, GMRES 40/43/55 vectors.
  - Plain: 3.7e-6 → 7.9e-9 → 7.0e-12, GMRES 42/41/53 vectors.
  - Floquet multipliers agree between the norms to 1e-7. Against SPECTER `floq_n100`, the 13 leading
    multipliers agree to 0.04–0.8 % (24.53 vs 24.47, |11.30| vs |11.28|, 5.458 vs 5.456, ...).
  - From the 14th on, n = 40 is **not converged** (native 1.0025 vs SPECTER 1.33): this orbit has 22
    unstable multipliers. Counting unstable directions needs `n` well above that count; check by
    comparing two values of `n`.
  - Default norm stays `quadrature`: same cost, and it is the physical norm, independent of the grid.

### `nu_sweep.py` (+ `job_nu_sweep.sh`, `nu_sweep_compare.py`): long-time Nu(Ra) from DNS

One `Ra` per run. Starts from the conduction state plus noise and integrates `--T` free-fall times
(default 1500, box `2 Lz x Lz` with `--Lz` default `pi`). Every `--chunk` time units (default 1) it
records diagnostics, then averages over `t > --tavg` (default 300). Error bars are block-average
standard errors (10 blocks).

| output | content |
|---|---|
| stdout | first chunk and every 50 chunks: `t`, the three Nu estimates, KE, CFL number, ms/step; final `SUMMARY` line |
| `<out>/ra<Ra>_<Nx>x<Nz>.npz` | time series `t, nu_flux, nu_grad, nu_wall, ke, umax` and `Ra, dt_ff, Lz, Nx, Nz` (saved progressively) |
| `<out>/summary.txt` | the `SUMMARY` lines appended |

**The `SUMMARY` line.**
- `Nu_flux = 1 + <w theta>/kappa`, `Nu_grad = <|grad T|^2>`, `Nu_wall` (wall heat flux), each with
  `+-` the standard error and `(std ...)` the spread of the time series.
- They must agree for long runs. A gap means the run is too short or under-resolved.
- `KE` and `umax` are in units of the free-fall velocity `sqrt(g alpha dT H)`: KE divided by `Lz^2`
  and umax by `Lz`. That makes them independent of `Lz` and comparable with SPECTER
  (`specter_nu_series.py`) and Dedalus.
- The appended `SPECTER DNS ref (gradient)` comes from the paper's CSV, which is **~6–10% high**:
  its `np.gradient` estimator over-weights the wall rows. Compare with the long-time SPECTER flux
  values instead.

**`nu_sweep_compare.py <out dirs>`.**
- Prints one table row per Ra: the three Nu estimates, KE, umax, the CSV value and the % difference,
  and the long-time SPECTER flux Nu for each SPECTER run in `../specter_series/`.
- Also lists the Dedalus reference values; `--fig` plots it all.
- Runs with fewer than 10 samples after `tavg` are skipped.

**Earlier results** (diffusive solver, 256×97, 2026-09-22; native vs SPECTER long-time flux Nu):

| Ra | native Nu | SPECTER Nu | note |
|---|---|---|---|
| 1e5 | 4.9943 | — | steady |
| 2e5 | 5.9721 | — | steady |
| 3e5 | 6.616 | — | steady |
| 9e5 | 7.499 | 7.500 | |
| 2e6 | 9.062 | 9.065 | Dedalus 9.054 |
| 5e6 | 11.38 | 11.38 | |
| 1e7 | 14.204 | 14.20 | |
| 2e7 | 16.94 | 17.02 | |

KE agrees within 1%. The free-fall rerun (2026-10-01, job 45107, `/share/data4/jcullen/rb_bench/item34/nusweep/`)
reproduces this table: 4.9943, 5.9721, 6.6146, 7.0712 (4e5), 7.4991, 7.5938 (1e6), 9.0564, 11.3784, 14.2007,
16.9385. Against the long-time SPECTER flux values it is within 0.0–0.1 %, except 4e5 (−0.4 %, slow
transient) and 5e6 at 256-103 (−0.3 %; the 256-231 run agrees to 0.0 %).

### `specter_nu_series.py`: long-time Nu and KE of a SPECTER run

```
python specter_nu_series.py <SPECTER run dir> [--tavg 300] [--stride 2] [--out series.npz]
```

Reads every field snapshot of a SPECTER DNS run and computes the flux Nu with the same definition as
`nu_sweep.py`. Ra is taken from the viscosity in `parameter.inp`, because `params.py` can be stale.
Prints averages over `t > tavg` and `t > 1000` (only with more than 20 snapshots).

`--out` saves `t, nu_flux, t_bal, ke_ff, Ra`, the format `nu_sweep_compare.py` reads from
`specter_series/`.
