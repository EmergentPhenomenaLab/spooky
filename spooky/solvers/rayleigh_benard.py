''' 2D Rayleigh-Benard convection solver.

Periodic in x (Fourier), wall-bounded in z (Chebyshev), primitive-variable
``(u, w, theta)`` formulation with an IMEX (implicit-explicit) time step:
diffusion and the incompressibility/pressure constraint are implicit, advection
and buoyancy are explicit.

Two time schemes are available. ``scheme='sbdf2'`` (default) is the
second-order semi-implicit BDF2:

    (3 f^{n+1} - 4 f^n + f^{n-1}) / (2 dt) = L f^{n+1} + 2 E(f^n) - E(f^{n-1})

with implicit coefficient lam = 3/(2 dt) and one backward-Euler startup step
(whose O(dt^2) local error preserves global second order). ``scheme='be'`` is
the original first-order backward-Euler / forward-Euler pair (lam = 1/dt).
Both use the same influence-matrix Stokes solve; only lam differs, so the
per-mode factorizations are precomputed once per scheme.

Non-dimensionalisation (free-fall units, layer height Lz = ``grid.Lz``):

    du/dt + u.grad u = -grad p + nu lap(u) + theta z_hat
    grad.u = 0
    dtheta/dt + u.grad theta = kappa lap(theta) + w

    nu = Lz^2 sqrt(Pr/Ra),   kappa = Lz^2 / sqrt(Pr Ra)

The time unit is the free-fall time sqrt(H / (g alpha DeltaT)) for any Lz; the
temperature difference across the layer is Lz (base gradient 1).  Lz = pi with
Lx = 2 pi is the SPECTER convention; Lz = 1 is the textbook free-fall scaling.
Solutions for different Lz map onto each other by rescaling x, z, u, w and
theta by the ratio of the Lz's (time unchanged).

``theta`` is the temperature perturbation about the linear conduction profile
T_cond = Lz - z (hot bottom, cold top), so theta = 0 at both walls.  Velocity
boundary conditions are no-slip (u=w=0) or free-slip (du/dz=0, w=0).

The pressure / incompressibility at the walls is handled with the classical
**influence-matrix** (Kleiser-Schumann) method.  Per Fourier mode the velocity
solve reduces to: one pressure-Poisson solve, two velocity Helmholtz solves, and
a 2x2 correction that picks the boundary pressure values so the velocity is both
divergence-free and satisfies the wall BC.  All operators are state-independent
and precomputed once (dense per-mode factorizations), which makes the whole step
a sequence of cached linear solves -- exactly differentiable under JAX.
'''
import os
from collections import OrderedDict
import numpy as np

from .._backend import xnp, index_update
from .imex import IMEXSpectral
from .. import pseudo as ps


def _apply_dirichlet(A, Nz):
    ''' Replace the two boundary rows of A by identity rows (Dirichlet). '''
    A = A.copy()
    A[0, :] = 0.0;  A[0, 0] = 1.0
    A[Nz - 1, :] = 0.0;  A[Nz - 1, Nz - 1] = 1.0
    return A


def _apply_neumann(A, D, Nz):
    ''' Replace the two boundary rows of A by the first-derivative (Neumann)
    rows taken from D. '''
    A = A.copy()
    A[0, :] = D[0, :]
    A[Nz - 1, :] = D[Nz - 1, :]
    return A


class RayleighBenard2D(IMEXSpectral):
    '''
    2D Rayleigh-Benard convection.

    Parameters
    ----------
    grid : Grid2D_wall
        Fourier-x / Chebyshev-z grid.
    ra : float
        Rayleigh number (may be changed after construction, e.g. by
        arclength continuation: the implicit operators are cached per
        (dt, ra, pr) and rebuilt on the next evolve).
    pr : float, optional
        Prandtl number. Default 1.0.
    bc : {'no-slip', 'free-slip'}, optional
        Velocity wall boundary condition. Default 'no-slip'.
    scheme : {'sbdf2', 'be'}, optional
        Time scheme: second-order SBDF2 (default) or first-order
        backward-Euler / forward-Euler.
    ftypes : list of str, optional
        Field names for output files.
    ext : int, optional
        Zero-padding length in output filenames.
    '''

    num_fields = 3
    dim_fields = 2

    def __init__(self,
                 grid: ps.Grid2D_wall,
                 ra,
                 pr=1.0,
                 bc='no-slip',
                 scheme='sbdf2',
                 ftypes=['uu', 'ww', 'th'],
                 ext=4):
        super().__init__(grid)
        if bc not in ('no-slip', 'free-slip'):
            raise ValueError("bc must be 'no-slip' or 'free-slip'")
        if scheme not in ('sbdf2', 'be'):
            raise ValueError("scheme must be 'sbdf2' or 'be'")
        self.solver = 'RayleighBenard2D'
        self.ra = ra
        self.pr = pr
        self.bc = bc
        self.scheme = scheme
        self.n_startup = 1 if scheme == 'sbdf2' else 0
        self.ftypes = ftypes
        self.ext = ext

        self._build_operators(grid.dt)

    @property
    def nu(self):
        ''' Kinematic viscosity in free-fall units, Lz^2 sqrt(Pr/Ra). '''
        return self.grid.Lz ** 2 * np.sqrt(self.pr / self.ra)

    @property
    def kappa(self):
        ''' Thermal diffusivity in free-fall units, Lz^2 / sqrt(Pr Ra). '''
        return self.grid.Lz ** 2 / np.sqrt(self.pr * self.ra)

    def ad_cache_key(self):
        ''' ad_evolve picks its implicit operators by (dt, ra, pr) at trace
        time, so compiled JVPs (DynSys.jvp_evolve) must be keyed on ra, pr. '''
        return (float(self.ra), float(self.pr))

    # ------------------------------------------------------------------
    # Precompute (numpy; state-independent operators)
    # ------------------------------------------------------------------
    def _build_operators(self, dt):
        grid = self.grid
        Nz = grid.Nz
        Nkx = grid.Nkx

        D = np.asarray(grid.Dz)
        D2 = np.asarray(grid.Dz2)
        Iz = np.eye(Nz)
        kx = np.asarray(grid.kx)

        # boundary-row zeroing operator (folds the homogeneous Dirichlet RHS)
        Z = np.eye(Nz)
        Z[0, 0] = 0.0
        Z[Nz - 1, Nz - 1] = 0.0
        e0 = np.zeros(Nz);  e0[0] = 1.0
        eN = np.zeros(Nz);  eN[Nz - 1] = 1.0

        # --- lam- and parameter-independent operators ---------------------
        # (real: L = D2 - k^2 I is real; see _solve for the complex apply)
        SLp = np.zeros((Nkx, Nz, Nz))                  # particular pressure-Poisson
        phi0 = np.zeros((Nkx, Nz))                     # wall-pressure influence fns
        phiN = np.zeros((Nkx, Nz))
        SProj = np.zeros((Nkx, Nz, Nz))                # Leray (Neumann) projection

        for i in range(Nkx):
            k = kx[i]
            L = D2 - (k ** 2) * Iz                      # Laplacian for this mode
            # pressure particular: Dirichlet p=0; influence fns: unit wall value
            Lp_inv = np.linalg.inv(_apply_dirichlet(L, Nz))
            SLp[i] = Lp_inv @ Z
            phi0[i] = Lp_inv @ e0
            phiN[i] = Lp_inv @ eN

            # --- Leray projection operator (no-penetration), for inc_proj ---
            # phi solves L phi = div(V) with Neumann dphi/dz = w at walls.
            # The k=0 Neumann Laplacian is singular (constant null space); that
            # mode is handled analytically in inc_proj (u unchanged, w=0), so
            # leave its operator as zero.
            if abs(k) > 1e-12:
                Ln_bc = _apply_neumann(L, D, Nz)
                SProj[i] = np.linalg.inv(Ln_bc)

        self._np_ops = dict(D=D, D2=D2, Iz=Iz, kx=kx, Z=Z, phi0=phi0, phiN=phiN)
        self.D = xnp.asarray(D)
        self.ik = 1j * grid.kx_col                      # (Nkx,1)
        self.SLp = xnp.asarray(SLp)
        self.SProj = xnp.asarray(SProj)

        # --- per-dt Stokes/Helmholtz operator sets (see _ops_for) --------
        self._ops_cache = OrderedDict()
        self._ops_for(dt)

    #: number of per-dt operator sets kept (each is 0.06 GB at 256x97 and
    #: 0.82 GB at 512x257 for SBDF2, on top of 0.02 / 0.27 GB of fixed SLp and
    #: SProj); a Newton run needs grid.dt, the current T and a few hookstep
    #: trial T's at once
    ops_cache_size = 4

    def _ops_for(self, dt):
        ''' Operator set for timestep ``dt``: ``{'be': ..., 'sbdf2': ...}`` with
        the backward-Euler set (the 'be' scheme and the SBDF2 startup step)
        and, for SBDF2, the bulk set.  Built in NumPy on first use and kept in
        a small LRU cache, since evolve() rescales dt to realise T exactly.
        The key includes ra and pr (through nu and kappa they enter the
        Helmholtz operators), so changing ``self.ra`` is picked up. '''
        key = (float(dt), float(self.ra), float(self.pr))
        cache = self._ops_cache
        if key not in cache:
            ops = {'be': self._build_stokes_ops(1.0 / key[0])}
            if self.scheme == 'sbdf2':
                ops['sbdf2'] = self._build_stokes_ops(1.5 / key[0])
            while len(cache) >= self.ops_cache_size:
                cache.popitem(last=False)
            cache[key] = ops
        cache.move_to_end(key)
        return cache[key]

    def _build_stokes_ops(self, lam):
        ''' Helmholtz solves and influence matrix for one implicit coefficient
        lam (velocity operator lam*I - nu*L, temperature lam*I - kappa*L). '''
        grid = self.grid
        Nz = grid.Nz
        Nkx = grid.Nkx
        nu, kappa = self.nu, self.kappa
        o = self._np_ops
        D, D2, Iz, kx, Z = o['D'], o['D2'], o['Iz'], o['kx'], o['Z']
        b0, b1 = 0, Nz - 1

        # Helmholtz solves are real (lam, nu, kappa, L real); only the
        # wall-pressure responses carry the factor ik
        SHu = np.zeros((Nkx, Nz, Nz))                  # u Helmholtz solve (BC per self.bc)
        SHw = np.zeros((Nkx, Nz, Nz))                  # w Helmholtz solve (Dirichlet)
        SHt = np.zeros((Nkx, Nz, Nz))                  # temperature Helmholtz solve
        Util = np.zeros((Nkx, 2, Nz), dtype=complex)   # u-response to wall pressure
        Wtil = np.zeros((Nkx, 2, Nz), dtype=complex)   # w-response to wall pressure
        Minv = np.zeros((Nkx, 2, 2), dtype=complex)    # inverse influence matrix

        for i in range(Nkx):
            k = kx[i]
            L = D2 - (k ** 2) * Iz                      # Laplacian for this mode
            Hv = lam * Iz - nu * L                      # velocity Helmholtz
            Ht = lam * Iz - kappa * L                   # temperature Helmholtz

            # u: no-slip -> Dirichlet (u=0); free-slip -> Neumann (du/dz=0).
            # Both are homogeneous, so the RHS boundary rows are zeroed (Z) either way.
            if self.bc == 'no-slip':
                Hu_bc = _apply_dirichlet(Hv, Nz)
            else:
                Hu_bc = _apply_neumann(Hv, D, Nz)
            # w: always Dirichlet (no penetration, w=0) in both BC types.
            Hw_bc = _apply_dirichlet(Hv, Nz)
            # temperature: Dirichlet (theta=0).
            Ht_bc = _apply_dirichlet(Ht, Nz)

            SHu[i] = np.linalg.inv(Hu_bc) @ Z
            SHw[i] = np.linalg.inv(Hw_bc) @ Z
            SHt[i] = np.linalg.inv(Ht_bc) @ Z

            if abs(k) < 1e-12:
                # k=0: continuity (dw/dz = 0) and w = 0 at the walls give w = 0
                # exactly, and the (hydrostatic) pressure does not act on u.  The
                # influence matrix is singular here (a uniform pressure offset has
                # no effect), so handle the mode analytically, as inc_proj does:
                # zero SHw, Util, Wtil and Minv, i.e. w = 0 and no correction.
                SHw[i] = 0.0
                continue

            # velocity response to the two wall-pressure modes (homogeneous wall BC)
            phi0 = o['phi0'][i];  phiN = o['phiN'][i]
            ut0 = SHu[i] @ (-1j * k * phi0);  wt0 = SHw[i] @ (-(D @ phi0))
            utN = SHu[i] @ (-1j * k * phiN);  wtN = SHw[i] @ (-(D @ phiN))
            Util[i, 0] = ut0;  Util[i, 1] = utN
            Wtil[i, 0] = wt0;  Wtil[i, 1] = wtN
            # influence matrix: wall divergence (ik*u + D*w) from each pressure mode
            div0 = 1j * k * ut0 + D @ wt0
            divN = 1j * k * utN + D @ wtN
            M = np.array([[div0[b0], divN[b0]],
                          [div0[b1], divN[b1]]], dtype=complex)
            Minv[i] = np.linalg.inv(M)

        return {name: xnp.asarray(arr) for name, arr in
                [('SHu', SHu), ('SHw', SHw), ('SHt', SHt),
                 ('Util', Util), ('Wtil', Wtil), ('Minv', Minv)]}

    # ------------------------------------------------------------------
    # Helpers (mixed space: Fourier in x, collocation in z)
    # ------------------------------------------------------------------
    @staticmethod
    def _solve(op, rhs):
        ''' Batched per-mode apply of a real operator op (Nkx,Nz,Nz) to a
        complex rhs (Nkx,Nz): one real batched matmul on [Re, Im] (half the
        flops and operator memory of a complex matmul; a plain einsum would
        promote op to complex). '''
        ri = xnp.stack([rhs.real, rhs.imag], axis=-1)          # (Nkx,Nz,2)
        out = op @ ri
        return out[..., 0] + 1j * out[..., 1]

    def _dz(self, fhat):
        return fhat @ self.D.T

    def _advection(self, uh, wh, th):
        ''' Explicit nonlinear advection terms (dealiased in x), mixed space. '''
        grid = self.grid
        u = grid.inverse(uh);   w = grid.inverse(wh);   t = grid.inverse(th)
        ux = grid.inverse(grid.ddx(uh));  uz = grid.inverse(self._dz(uh))
        wx = grid.inverse(grid.ddx(wh));  wz = grid.inverse(self._dz(wh))
        tx = grid.inverse(grid.ddx(th));  tz = grid.inverse(self._dz(th))
        a_u = grid.forward(u * ux + w * uz)
        a_w = grid.forward(u * wx + w * wz)
        a_t = grid.forward(u * tx + w * tz)
        # dealias in x (2/3 rule)
        mask = grid.dealias_modes
        a_u = xnp.where(mask, 0.0, a_u)
        a_w = xnp.where(mask, 0.0, a_w)
        a_t = xnp.where(mask, 0.0, a_t)
        return a_u, a_w, a_t

    # ------------------------------------------------------------------
    # IMEX stepping: explicit terms + implicit Stokes/temperature solve
    # ------------------------------------------------------------------
    def _explicit_terms(self, uh, wh, th):
        ''' Explicitly-treated terms E(f): advection, buoyancy, and the
        conduction-profile source w in the temperature equation. '''
        a_u, a_w, a_t = self._advection(uh, wh, th)
        return [-a_u,
                -a_w + th,
                -a_t + wh]

    def _imex_solve(self, ops, r_u, r_w, r_t):
        ''' Implicit solve for one step: temperature Helmholtz + velocity
        Stokes via the influence matrix, given the full RHS r. '''
        ik = self.ik

        # temperature Helmholtz solve (theta = 0 at walls)
        th_new = self._solve(ops['SHt'], r_t)

        # velocity Stokes solve via influence matrix
        rhs_p = ik * r_u + self._dz(r_w)
        p_p = self._solve(self.SLp, rhs_p)
        u_p = self._solve(ops['SHu'], r_u - ik * p_p)
        w_p = self._solve(ops['SHw'], r_w - self._dz(p_p))

        # wall divergence residual (general: ik*u + D*w at the two walls)
        div = ik * u_p + self._dz(w_p)
        d = xnp.stack([div[:, 0], div[:, -1]], axis=-1)      # (Nkx,2)
        c = -xnp.einsum('mij,mj->mi', ops['Minv'], d)        # (Nkx,2)
        u_new = u_p + c[:, 0:1] * ops['Util'][:, 0] + c[:, 1:2] * ops['Util'][:, 1]
        w_new = w_p + c[:, 0:1] * ops['Wtil'][:, 0] + c[:, 1:2] * ops['Wtil'][:, 1]

        # 2/3-rule dealiasing in x (each x-mode is independent, so this preserves
        # the wall BCs and the divergence-free property of the retained modes)
        mask = self.grid.dealias_modes
        u_new = xnp.where(mask, 0.0, u_new)
        w_new = xnp.where(mask, 0.0, w_new)
        th_new = xnp.where(mask, 0.0, th_new)

        return [u_new, w_new, th_new]

    def imex_step(self, fields, dt, ops=None):
        ''' First-order backward-Euler / forward-Euler IMEX step. '''
        if ops is None:
            ops = self._ops_for(dt)
        E = self._explicit_terms(*fields)
        r = [f / dt + e for f, e in zip(fields, E)]
        return self._imex_solve(ops['be'], *r)

    # ------------------------------------------------------------------
    # SBDF2 state protocol: carry = (f^n, f^{n-1}, E^{n-1})
    # ------------------------------------------------------------------
    def init_state(self, fields, dt, ops=None):
        ''' Build the scheme carry; SBDF2 takes one backward-Euler startup
        step (n_startup = 1) to fill the second time level. '''
        if self.scheme == 'be':
            return fields
        if ops is None:
            ops = self._ops_for(dt)
        E0 = self._explicit_terms(*fields)
        r = [f / dt + e for f, e in zip(fields, E0)]
        f1 = self._imex_solve(ops['be'], *r)
        return (f1, list(fields), E0)

    def step_state(self, state, dt, ops=None):
        if self.scheme == 'be':
            return self.imex_step(state, dt, ops=ops)
        if ops is None:
            ops = self._ops_for(dt)
        fn, fnm1, Enm1 = state
        En = self._explicit_terms(*fn)
        r = [2.0 / dt * a - 0.5 / dt * b + 2.0 * e - ep
             for a, b, e, ep in zip(fn, fnm1, En, Enm1)]
        fnew = self._imex_solve(ops['sbdf2'], *r)
        return (fnew, list(fn), En)

    def state_fields(self, state):
        return state if self.scheme == 'be' else state[0]

    # ------------------------------------------------------------------
    # Solenoidal (Leray, no-penetration) projection -- for DynSys.sol_project
    # ------------------------------------------------------------------
    def inc_proj(self, fields):
        ''' Project a physical velocity (and pass temperature through) onto the
        divergence-free, no-penetration subspace. '''
        grid = self.grid
        uh = grid.forward(fields[0])
        wh = grid.forward(fields[1])
        div = self.ik * uh + self._dz(wh)
        # Neumann RHS: boundary rows carry the wall-normal velocity values, so
        # that dphi/dz = w at the walls and the corrected w vanishes there.
        rhs = index_update(div, (slice(None), 0), wh[:, 0])
        rhs = index_update(rhs, (slice(None), -1), wh[:, -1])
        phi = self._solve(self.SProj, rhs)
        u_proj = uh - self.ik * phi
        w_proj = wh - self._dz(phi)
        # k=0 mode: incompressibility forces w=0; u (x-mean) is unconstrained.
        u_proj = index_update(u_proj, 0, uh[0])
        w_proj = index_update(w_proj, 0, 0.0)
        # restrict to the resolved (2/3-dealiased) x-subspace; this also drops the
        # rfft Nyquist mode, whose phase-shift correction irfft cannot represent.
        u_proj = xnp.where(grid.dealias_modes, 0.0, u_proj)
        w_proj = xnp.where(grid.dealias_modes, 0.0, w_proj)
        u_new = grid.inverse(u_proj)
        w_new = grid.inverse(w_proj)
        out = [u_new, w_new]
        if len(fields) > 2:
            out.append(fields[2])
        return out

    # ------------------------------------------------------------------
    # Diagnostics / IO
    # ------------------------------------------------------------------
    def nusselt(self, fields):
        ''' Volume-averaged Nusselt number Nu = 1 + <w theta> / kappa. '''
        u, w, t = fields
        return 1.0 + self.grid.average(w * t) / self.kappa

    def balance(self, fields, step, bpath):
        u, w, t = [self.grid.inverse(f) if xnp.iscomplexobj(f) else f for f in fields]
        ke = 0.5 * self.grid.average(u ** 2 + w ** 2)
        nu = self.nusselt([u, w, t])
        bal = [f'{self.grid.dt*step:.4e}', f'{float(ke):.6e}', f'{float(nu):.6e}']
        with open(os.path.join(bpath, 'balance.dat'), 'a') as output:
            print(*bal, file=output)

    def outs(self, fields, step, opath):
        u, w, t = [self.grid.inverse(f) if xnp.iscomplexobj(f) else f for f in fields]
        for name, fld in zip(self.ftypes, [u, w, t]):
            np.save(os.path.join(opath, f'{name}.{step:0{self.ext}}'), np.asarray(fld))
