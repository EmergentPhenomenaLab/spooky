"""Tests for the SPECTER <-> native RB solver grid conversion."""
import numpy as np
import pytest
import spooky as sp
from spooky import rb_convert as rc


def _analytic(Nx, Nz, Lx, Lz):
    x = np.linspace(0, Lx, Nx, endpoint=False)[:, None]
    z = np.linspace(0, Lz, Nz)[None, :]
    f = np.sin(2 * np.pi * x / Lx) * np.sin(np.pi * z / Lz) ** 2 + 0.3 * np.cos(4 * np.pi * x / Lx) * (z / Lz) * (1 - z / Lz)
    return f


def test_resample_x_roundtrip():
    f = _analytic(64, 9, 2 * np.pi, np.pi)
    g = rc.resample_x(rc.resample_x(f, 128), 64)
    np.testing.assert_allclose(g, f, atol=1e-12)


def test_uniform_cheb_roundtrip():
    grid = sp.Grid2D_wall(2.0, 1.0, 16, 33, 1e-4)
    f = _analytic(16, 257, 2.0, 1.0)                       # uniform, walls included
    fc = rc.uniform_to_cheb(f, np.asarray(grid.zi), Lz_in=1.0)
    fu = rc.cheb_to_uniform(fc, np.asarray(grid.zi), 257, Lz_out=1.0)
    assert np.max(np.abs(fu - f)) < 1e-7                  # quintic spline on 257 pts


def test_cheb_to_uniform_exact_on_polynomial():
    grid = sp.Grid2D_wall(2.0, 1.0, 4, 17, 1e-4)
    z = np.asarray(grid.zi)
    f = np.tile((z ** 5 - 0.3 * z ** 2 + 1.0)[None, :], (4, 1))
    zu = np.linspace(0, 1, 41)
    fu = rc.cheb_to_uniform(f, z, 41)
    np.testing.assert_allclose(fu, np.tile((zu ** 5 - 0.3 * zu ** 2 + 1.0)[None, :], (4, 1)), atol=1e-12)


def _fields(x, z, Lx_s=2 * np.pi, Lz_s=np.pi):
    """Analytic SPECTER-like fields at SPECTER coordinates (vanish at walls)."""
    x, z = x[:, None], z[None, :]
    w_s = 0.8 * np.sin(2 * np.pi * x / Lx_s) * np.sin(np.pi * z / Lz_s)
    u_s = -0.8 * (Lx_s / (2 * Lz_s)) * np.cos(2 * np.pi * x / Lx_s) * np.sin(2 * np.pi * z / Lz_s) / 2
    th_s = 0.5 * np.sin(2 * np.pi * x / Lx_s) * np.sin(np.pi * z / Lz_s) + 0.2 * np.sin(2 * np.pi * z / Lz_s)
    return [u_s, w_s, th_s]


def _specter_like(Nx_s=64, Nz_s=129, Lx_s=2 * np.pi, Lz_s=np.pi):
    """The same on a SPECTER grid (uniform z including both walls)."""
    return _fields(np.linspace(0, Lx_s, Nx_s, endpoint=False), np.linspace(0, Lz_s, Nz_s), Lx_s, Lz_s)


@pytest.mark.parametrize('Lz', [np.pi, 1.0])
def test_specter_roundtrip_and_nusselt(Lz):
    """SPECTER box -> native grid of height Lz and back.  Lz = pi: same units,
    interpolation only; Lz = 1: fields scaled by 1/pi.  Nu is the same."""
    from spooky.solvers import RayleighBenard2D
    ra, pr = 1e6, 1.0
    fs = _specter_like()
    grid = sp.Grid2D_wall(2.0 * Lz, Lz, 64, 33, 1e-3)
    solver = RayleighBenard2D(grid, ra=ra, pr=pr)
    fn = rc.specter_to_native(fs, grid)
    f = rc.length_factor(grid)
    assert f == pytest.approx(Lz / np.pi)

    # expected: f times the analytic fields at the native nodes mapped to SPECTER coordinates
    x_n = np.linspace(0, grid.Lx, grid.Nx, endpoint=False)
    expected = [f * e for e in _fields(x_n / f, np.asarray(grid.zi) / f)]
    for a, e in zip(fn, expected):
        assert np.max(np.abs(a - e)) < 1e-7 * np.max(np.abs(e))
        assert np.max(np.abs(a[:, [0, -1]])) == 0.0

    # Nusselt is unit-independent (trapezoid on 129 uniform points vs Clenshaw-Curtis)
    nu_s = rc.specter_nusselt(fs, ra, pr)
    nu_n = float(solver.nusselt(fn))
    assert abs(nu_n - nu_s) < 1e-4 * nu_s

    back = rc.native_to_specter(fn, grid, Nx_out=64, Nz_out=129)
    for a, b in zip(back, fs):
        assert np.max(np.abs(a - b)) < 1e-5 * max(1.0, np.max(np.abs(b)))

    sx_s = 0.7
    assert rc.shift_to_native(sx_s, grid) == pytest.approx(f * sx_s, rel=1e-14)
    assert rc.shift_to_specter(rc.shift_to_native(sx_s, grid), grid) == pytest.approx(sx_s, rel=1e-14)


def test_specter_to_native_rejects_other_aspect_ratio():
    grid = sp.Grid2D_wall(3.0, 1.0, 16, 17, 1e-3)
    with pytest.raises(ValueError, match='aspect ratio'):
        rc.specter_to_native(_specter_like(16, 17), grid)


def test_quadrature_weights_give_rms_norm():
    """DynSys(solver, weights=quadrature_weights(grid)): |U|^2 = sum_i <f_i^2>."""
    from spooky.solvers import RayleighBenard2D
    from spooky.methods import DynSys
    grid = sp.Grid2D_wall(2 * np.pi, np.pi, 32, 25, 1e-3)
    ds = DynSys(RayleighBenard2D(grid, ra=1e5), weights=rc.quadrature_weights(grid))
    fields = rc.specter_to_native(_specter_like(), grid)
    U = ds.flatten(fields)
    np.testing.assert_allclose(np.dot(U, U), sum(float(grid.average(f ** 2)) for f in fields),
                               rtol=1e-12)
