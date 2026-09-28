Examples and use cases.

- test_derivative: tests derivatives in 1- and 2D
- kse: forward evolution of the 1D Kuramoto-Sivashinsky equation
- kolmogorov: forward evolution of 2D Kolmogorov flow
- newton_kolmog: Newton-Krylov-hookstep search for a UPO of Kolmogorov flow (orb05)
- ad_solver: JAX backend only — gradients through the solver, JVP vs finite-difference
  tangent accuracy, Floquet/Lyapunov with exact tangents, gradient-descent UPO seeding
- benchmarks: per-step cost / memory of the AD paths (forward, JVP, gradient)
- specter_wrapper, ghost_wrapper: time marching and Lyapunov runs through the Fortran codes
