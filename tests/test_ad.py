"""Tests for the AD (auto-differentiable) solver pipeline.

Run with:  NUMPY_BACKEND=jax pytest tests/test_ad.py
"""
import os
import pytest
import numpy as np

# Skip entire module when not running under JAX backend
pytestmark = pytest.mark.skipif(
    os.environ.get('NUMPY_BACKEND', 'numpy').lower() != 'jax',
    reason="AD tests require NUMPY_BACKEND=jax",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def kse_setup():
    import spooky as sp
    from spooky.solvers import KuramotoSivashinsky
    grid   = sp.Grid1D(22.0, 64, 1e-3)
    solver = KuramotoSivashinsky(grid, nu=1.0, rkord=2)
    return grid, solver


@pytest.fixture(scope='module')
def kf_setup():
    import spooky as sp
    from spooky.solvers import KolmogorovFlow
    grid   = sp.Grid2D(2 * np.pi, 2 * np.pi, 32, 32, 0.01)
    solver = KolmogorovFlow(grid, kf=4, f0=1.0, nu=0.1, rkord=2)
    return grid, solver


def _kse_ic(grid):
    Lx = grid.Lx
    return (0.3 * np.cos(2 * np.pi * 3 * grid.xx / Lx) +
            0.4 * np.cos(2 * np.pi * 5 * grid.xx / Lx) +
            0.5 * np.cos(2 * np.pi * 4 * grid.xx / Lx))


def _kf_ic(grid, seed=42):
    rng = np.random.default_rng(seed)
    uu  = rng.standard_normal(grid.shape)
    vv  = rng.standard_normal(grid.shape)
    fu, fv = grid.inc_proj([grid.forward(uu), grid.forward(vv)])
    return grid.inverse(fu), grid.inverse(fv)


# ---------------------------------------------------------------------------
# Trajectory correctness
# ---------------------------------------------------------------------------

def test_ad_evolve_matches_evolve_kse(kse_setup):
    """ad_evolve and evolve must produce identical trajectories (KSE, 1D)."""
    import jax.numpy as jnp
    grid, solver = kse_setup
    ic = _kse_ic(grid)

    ref = solver.evolve([ic], T=0.1, write_outputs=False)
    out = solver.ad_evolve([jnp.array(ic)], T=0.1)

    np.testing.assert_allclose(np.array(out[0]), ref[0], atol=1e-10,
                               err_msg="KSE: ad_evolve differs from evolve")


def test_ad_evolve_matches_evolve_kolmogorov(kf_setup):
    """ad_evolve and evolve must produce identical trajectories (Kolmogorov, 2D)."""
    import jax.numpy as jnp
    grid, solver = kf_setup
    uu, vv = _kf_ic(grid)

    ref = solver.evolve([uu, vv], T=0.1, write_outputs=False)
    out = solver.ad_evolve([jnp.array(uu), jnp.array(vv)], T=0.1)

    np.testing.assert_allclose(np.array(out[0]), ref[0], atol=1e-10)
    np.testing.assert_allclose(np.array(out[1]), ref[1], atol=1e-10)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_grad_compiles_and_is_finite_kse(kse_setup):
    """jax.grad through ad_evolve must compile and return a finite gradient."""
    import jax
    import jax.numpy as jnp
    grid, solver = kse_setup
    u0 = jnp.array(_kse_ic(grid))

    def loss(u):
        out = solver.ad_evolve([u], T=0.1)
        return jnp.sum(out[0] ** 2)

    grad = jax.grad(loss)(u0)
    assert grad.shape == u0.shape
    assert jnp.all(jnp.isfinite(grad)), "Gradient contains non-finite values"


def test_grad_compiles_and_is_finite_kolmogorov(kf_setup):
    """jax.grad through ad_evolve must compile and return a finite gradient."""
    import jax
    import jax.numpy as jnp
    grid, solver = kf_setup
    uu, vv = _kf_ic(grid)
    u0 = jnp.array(uu)
    v0 = jnp.array(vv)

    def loss(u):
        out = solver.ad_evolve([u, v0], T=0.1)
        return jnp.sum(out[0] ** 2) + jnp.sum(out[1] ** 2)

    grad = jax.grad(loss)(u0)
    assert grad.shape == u0.shape
    assert jnp.all(jnp.isfinite(grad)), "Gradient contains non-finite values"


# ---------------------------------------------------------------------------
# JVP accuracy: AD tangent vs finite-difference tangent
# ---------------------------------------------------------------------------

def test_jvp_matches_fd_kse(kse_setup):
    """DynSys.jvp_evolve tangent must agree with finite-difference tangent (KSE)."""
    import jax.numpy as jnp
    from spooky.methods import DynSys

    grid, solver = kse_setup
    ds = DynSys(solver)

    u0   = jnp.array(_kse_ic(grid))
    du0  = jnp.array(np.random.default_rng(0).standard_normal(u0.shape) * 1e-5)
    T    = 0.1

    _, tangent_ad = ds.jvp_evolve([u0], [du0], T)

    base  = solver.ad_evolve([u0], T)
    pert  = solver.ad_evolve([u0 + du0], T)
    tangent_fd = [pert[0] - base[0]]

    np.testing.assert_allclose(
        np.array(tangent_ad[0]), np.array(tangent_fd[0]),
        rtol=1e-4, atol=1e-8,
        err_msg="KSE: AD and FD tangents disagree",
    )


def test_jvp_matches_fd_kolmogorov(kf_setup):
    """DynSys.jvp_evolve tangent must agree with finite-difference tangent (Kolmogorov)."""
    import jax.numpy as jnp
    from spooky.methods import DynSys

    grid, solver = kf_setup
    ds = DynSys(solver)

    uu, vv = _kf_ic(grid)
    rng    = np.random.default_rng(7)
    u0     = jnp.array(uu)
    v0     = jnp.array(vv)
    du0    = jnp.array(rng.standard_normal(uu.shape) * 1e-5)
    dv0    = jnp.array(rng.standard_normal(vv.shape) * 1e-5)
    T      = 0.1

    _, tangents_ad = ds.jvp_evolve([u0, v0], [du0, dv0], T)

    base = solver.ad_evolve([u0, v0], T)
    pert = solver.ad_evolve([u0 + du0, v0 + dv0], T)
    tangents_fd = [pert[i] - base[i] for i in range(2)]

    for i, (t_ad, t_fd) in enumerate(zip(tangents_ad, tangents_fd)):
        np.testing.assert_allclose(
            np.array(t_ad), np.array(t_fd),
            rtol=1e-4, atol=1e-8,
            err_msg=f"Kolmogorov field {i}: AD and FD tangents disagree",
        )


# ---------------------------------------------------------------------------
# VJP: adjoint consistency check (J^T·v, J·w inner product test)
# ---------------------------------------------------------------------------

def test_vjp_adjoint_consistency_kolmogorov(kf_setup):
    """<J·w, v> == <w, J^T·v> to machine precision (Kolmogorov)."""
    import jax.numpy as jnp
    from spooky.methods import DynSys

    grid, solver = kf_setup
    ds = DynSys(solver)

    uu, vv  = _kf_ic(grid)
    rng     = np.random.default_rng(13)
    u0, v0  = jnp.array(uu), jnp.array(vv)
    w       = [jnp.array(rng.standard_normal(uu.shape)),
               jnp.array(rng.standard_normal(vv.shape))]
    z       = [jnp.array(rng.standard_normal(uu.shape)),
               jnp.array(rng.standard_normal(vv.shape))]
    T       = 0.1

    _, Jw   = ds.jvp_evolve([u0, v0], w, T)
    _, JTz  = ds.vjp_evolve([u0, v0], z, T)

    # <J·w, z>
    lhs = sum(float(jnp.sum(Jw[i] * z[i]))  for i in range(2))
    # <w, J^T·z>
    rhs = sum(float(jnp.sum(w[i] * JTz[0][i])) for i in range(2))

    np.testing.assert_allclose(lhs, rhs, rtol=1e-6,
                               err_msg="Adjoint consistency <Jw,z> != <w,J^Tz>")
