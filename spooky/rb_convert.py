'''Grid conversion between the native RB solver and SPECTER/BOUSS.

Both codes use free-fall units: buoyancy coefficient g*alpha = 1, unit base
temperature gradient (Delta T = layer height), time = free-fall time, and
nu = Lz^2 sqrt(Pr/Ra), kappa = Lz^2/sqrt(Pr Ra).  SPECTER runs use the box
[2*pi, pi] (Lz_s = pi).  The native :class:`~spooky.solvers.RayleighBenard2D`
takes its layer height from ``grid.Lz``; with ``grid.Lz = pi`` the two codes
share units and only the grids differ.  For any other ``grid.Lz`` the same
physical state is a pure change of length unit (subscript n = native,
s = SPECTER, f = grid.Lz / Lz_s):

    x_n = f x_s,  z_n = f z_s,  sx_n = f sx_s
    u_n = f u_s,  theta_n = f theta_s                 (velocity and Delta T scale with length)
    t_n = t_s                                          (periods, dt, growth rates unchanged)
    Nu unchanged

Grids: SPECTER fields are on a uniform z grid that *includes both walls*
(rows 0 and N-1 are exactly the walls; spacing Lz_s/(N-1)) -- verified on the
converged BOUSS binaries, and different from ``Grid2D_semi`` (endpoint=False).
The native solver uses Chebyshev-Gauss-Lobatto nodes.  Interpolation uniform ->
Chebyshev uses a quintic spline in z (adequate for Newton initial guesses);
Chebyshev -> uniform uses barycentric interpolation (spectrally exact).  x is
resampled by Fourier truncation / zero padding when Nx differs.
'''
import os
import numpy as np
from scipy.interpolate import make_interp_spline

SPECTER_LZ = np.pi
SPECTER_LX = 2 * np.pi


# ----------------------------------------------------------------------------
# Scalars
# ----------------------------------------------------------------------------

def length_factor(grid, Lz_s=SPECTER_LZ):
    ''' f = grid.Lz / Lz_s: native lengths, velocities and theta are f times
    SPECTER's (1 for the SPECTER box, grid.Lz = pi). '''
    return float(grid.Lz) / Lz_s


def shift_to_native(sx_s, grid, Lz_s=SPECTER_LZ):
    return sx_s * length_factor(grid, Lz_s)


def shift_to_specter(sx_n, grid, Lz_s=SPECTER_LZ):
    return sx_n / length_factor(grid, Lz_s)


def specter_nu_kappa(ra, pr=1.0, Lz_s=SPECTER_LZ, gamma=1.0):
    ''' nu, kappa as written by the SPECTER wrapper for given Ra, Pr. '''
    nu = gamma * Lz_s ** 2 * np.sqrt(pr / ra)
    kappa = gamma * Lz_s ** 2 / np.sqrt(pr * ra)
    return nu, kappa


# ----------------------------------------------------------------------------
# Binary I/O (SPECTER/BOUSS field files, Fortran order)
# ----------------------------------------------------------------------------

def load_specter_fields(path, idx, shape, ftypes=('vx', 'vz', 'th'), ext=5,
                        precision='double'):
    ''' Load ``{ftype}.{idx:0{ext}}.out`` files of shape (Nx, Nz). '''
    dtype = np.float64 if precision == 'double' else np.float32
    out = []
    for ft in ftypes:
        f = os.path.join(path, f'{ft}.{idx:0{ext}}.out')
        out.append(np.fromfile(f, dtype=dtype).reshape(shape, order='F'))
    return out


# ----------------------------------------------------------------------------
# Grid interpolation
# ----------------------------------------------------------------------------

def resample_x(field, Nx_out):
    ''' Fourier resampling along axis 0 (periodic direction). '''
    Nx_in = field.shape[0]
    if Nx_out == Nx_in:
        return np.asarray(field)
    fh = np.fft.rfft(np.asarray(field), axis=0)
    Nk_out = Nx_out // 2 + 1
    fh_out = np.zeros((Nk_out,) + fh.shape[1:], dtype=complex)
    nk = min(Nk_out, fh.shape[0])
    fh_out[:nk] = fh[:nk]
    if Nx_out < Nx_in and Nx_out % 2 == 0:
        fh_out[-1] = 0.0                    # drop the new Nyquist mode
    return np.fft.irfft(fh_out, n=Nx_out, axis=0) * (Nx_out / Nx_in)


def uniform_to_cheb(field, z_cheb, Lz_in=1.0, k=5):
    ''' Interpolate along axis 1 from a uniform grid including both walls
    (z_j = j*Lz_in/(N-1)) to nodes ``z_cheb`` on the same interval [0, Lz_in]. '''
    field = np.asarray(field)
    N = field.shape[1]
    z_in = np.linspace(0.0, Lz_in, N)
    spl = make_interp_spline(z_in, field, k=k, axis=1)
    return spl(np.asarray(z_cheb))


def cheb_to_uniform(field, z_cheb, Nz_out, Lz_out=1.0):
    ''' Barycentric interpolation along axis 1 from Chebyshev-Gauss-Lobatto
    nodes ``z_cheb`` (on [0, Lz]) to a uniform grid of Nz_out points including
    both walls of the same interval scaled to [0, Lz_out]. '''
    field = np.asarray(field)
    z_cheb = np.asarray(z_cheb)
    N = len(z_cheb) - 1
    # barycentric weights of Chebyshev-Gauss-Lobatto nodes: (-1)^j, halved at the ends
    w = np.array([(-1.0) ** j for j in range(N + 1)]); w[0] *= 0.5; w[-1] *= 0.5
    Lz_in = z_cheb.max() - z_cheb.min()
    z_out = np.linspace(0.0, Lz_out, Nz_out) * (Lz_in / Lz_out) + z_cheb.min()
    out = np.zeros(field.shape[:1] + (Nz_out,))
    for i, z in enumerate(z_out):
        hit = np.where(np.abs(z - z_cheb) < 1e-14 * max(1.0, Lz_in))[0]
        if hit.size:
            out[:, i] = field[:, hit[0]]
            continue
        c = w / (z - z_cheb)
        out[:, i] = (field @ c) / c.sum()
    return out


# ----------------------------------------------------------------------------
# Field conversion
# ----------------------------------------------------------------------------

def _check_aspect(grid, Lx_s, Lz_s):
    if not np.isclose(grid.Lx / grid.Lz, Lx_s / Lz_s, rtol=1e-12):
        raise ValueError(f'aspect ratio mismatch: native Lx/Lz = {grid.Lx / grid.Lz}, '
                         f'SPECTER Lx/Lz = {Lx_s / Lz_s}')


def specter_to_native(fields_s, grid, Lx_s=SPECTER_LX, Lz_s=SPECTER_LZ):
    ''' SPECTER fields [vx, vz, th] (uniform z incl. walls, box [Lx_s, Lz_s])
    to native fields [u, w, theta] on ``grid`` (a Grid2D_wall with the same
    aspect ratio): interpolation, plus the length factor grid.Lz/Lz_s when
    grid.Lz != Lz_s. '''
    _check_aspect(grid, Lx_s, Lz_s)
    fac = length_factor(grid, Lz_s)
    z_target = np.asarray(grid.zi)                          # nodes on [0, grid.Lz]
    out = []
    for f in fields_s:
        f = resample_x(f, grid.Nx)
        f = uniform_to_cheb(f, z_target, Lz_in=grid.Lz)
        out.append(fac * f)
    # enforce the homogeneous wall values exactly
    for f in out:
        f[:, 0] = 0.0; f[:, -1] = 0.0
    return out


def native_to_specter(fields_n, grid, Nx_out=None, Nz_out=None, Lz_s=SPECTER_LZ):
    ''' Inverse of :func:`specter_to_native`: native fields on ``grid`` to
    SPECTER fields on a uniform z grid (both walls) of Nz_out points. '''
    Nx_out = Nx_out or grid.Nx
    Nz_out = Nz_out or grid.Nz
    fac = 1.0 / length_factor(grid, Lz_s)
    out = []
    for f in fields_n:
        f = cheb_to_uniform(np.asarray(f), np.asarray(grid.zi), Nz_out, Lz_out=grid.Lz)
        f = resample_x(f, Nx_out)
        out.append(fac * f)
    return out


# ----------------------------------------------------------------------------
# Norms and diagnostics
# ----------------------------------------------------------------------------

def quadrature_weights(grid, num_fields=3):
    ''' Per-field ``DynSys`` weights sqrt(dx w_z / (Lx Lz)) (uniform x,
    Clenshaw-Curtis in z; interior rows on wall-bounded grids), so that the
    Euclidean norm of the flat vector is the physical root-mean-square norm. '''
    wz = np.asarray(grid.wz)
    if getattr(grid, 'remove_boundary', False):
        wz = wz[1:-1]
    q = np.sqrt(grid.dx * wz / (grid.Lx * grid.Lz))[None, :]
    return [q] * num_fields


def specter_nusselt(fields_s, ra, pr=1.0, Lz_s=SPECTER_LZ):
    ''' Nu = 1 + <w th>/(kappa * dT/dz) with SPECTER's kappa and unit base
    gradient, volume-averaged with the trapezoid rule in z (walls included). '''
    _, w_s, th_s = fields_s
    _, kappa = specter_nu_kappa(ra, pr, Lz_s)
    wt = (w_s * th_s).mean(axis=0)
    N = wt.shape[0]
    integral = np.trapezoid(wt, dx=Lz_s / (N - 1)) if hasattr(np, 'trapezoid') else np.trapz(wt, dx=Lz_s / (N - 1))
    return 1.0 + integral / Lz_s / kappa
