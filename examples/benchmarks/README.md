Benchmarks of the native solvers against SPECTER and Dedalus.

- `bench_kolmogorov.py` — per-step cost of `KolmogorovFlow`: under NumPy `solver.evolve`;
  under JAX the jitted `ad_evolve` forward map, one JVP and one gradient through it, and
  `solver.evolve`. `--dtype float32` for the single-precision comparison.
- `bench_rb.py` — the same for `RayleighBenard2D` (free-fall units, dt = 1e-3); `--evolve` also
  times `solver.evolve`, `--mem` reports the compiled memory (reverse tape). With
  `NUMPY_BACKEND=numpy` it times `solver.evolve` only.
- `bench_dedalus_rb.py` — Dedalus v3 on the same RB problem, fixed `dt`, no analysis
  (run inside the `dedalus3` micromamba environment with `mpiexec`).
- `job_bench_cpu.sh` — one exclusive CPU node (default `c3`, where the SPECTER tables of
  `specter_runs/bench` were measured): native thread scaling, NumPy, Kolmogorov, Dedalus MPI scaling.
- `job_bench_gpu.sh` — one GPU (default an L4 on `c6`; `sbatch -w c4` for an A100): native RB,
  Kolmogorov, and `examples/rayleigh_benard/specter_orbit_dt_study.py`.
- `compare.py` — merges the job outputs with the SPECTER tables into one table and figure.

Override the environment with `SPOOKY_ENV=... sbatch job_bench_gpu.sh`. The raw outputs of the
2026-09-21 runs (diffusive-units solver, before the real-valued operators) are kept in the
`AD-diffusive` branch under `examples/benchmarks/results/`.
