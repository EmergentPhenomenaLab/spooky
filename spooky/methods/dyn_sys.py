from .krylov import GMRES, backsub, arnoldi_eig
import os
import numpy as np
import functools

from .. import pseudo as ps

class DynSys():
    """General class for Dynamical Systems-based methods.

    Provides decorated version of several functions from grid and solver that can work with a flattened field.

    Methods implemented:
        - Floquet multipliers
        - Lyapunov exponents

    Parameters
    ----------
    solver: Solver instance
        Solver to be used.
    """
    def __init__(self, solver):
        self.solver = solver
        self.grid = solver.grid

        self.remove_boundary = False
        if isinstance(self.grid, ps.Grid2D_semi):
            self.remove_boundary = True

    def flatten(self, fields):
        '''Flattens fields'''
        if not self.remove_boundary:
            return np.concatenate([f.flatten() for f in fields])
        else:
            return np.concatenate([f[:,1:-1].flatten() for f in fields])

    def unflatten(self, U):
        '''Unflatten fields'''
        ll = len(U)//self.solver.num_fields
        fields = [U[i*ll:(i+1)*ll] for i in range(self.solver.num_fields)]
        if not self.remove_boundary:
            fields = [f.reshape(self.grid.shape) for f in fields]
            return fields
        else:
            trim_shape = (self.grid.Nx, self.grid.Nz-2)
            trim_fields = [f.reshape(trim_shape) for f in fields]
            fields = [np.pad(f, pad_width = ((0,0),(1,1))) for f in trim_fields] #pads second dimension with zeros before and after
            return fields

    def flatten_dec(func):
        """Decorator that allows to work with flattened fields U instead of fields"""
        @functools.wraps(func)
        def wrapper(self, U, *args, **kwargs):
            fields = self.unflatten(U)
            result = func(self, fields, *args, **kwargs)
            return self.flatten(result)
        return wrapper

    @flatten_dec
    def evolve(self, U, *args, **kwargs):
        '''Evolves fields U in time T'''
        return self.solver.evolve(U, *args, **kwargs)

    @flatten_dec
    def translate(self, U, *args):
        '''Translates fields by sx in x direction'''
        return self.grid.translate(U, *args)

    @flatten_dec
    def deriv_U(self, U, *args):
        '''Derivatives in x direction of fields'''
        return self.grid.deriv_fields(U, *args)

    @flatten_dec
    def sol_project(self, U):
        '''Solenoidal projection of fields'''
        return self.solver.inc_proj(U)

    def write_fields(self, U, idx, path):
        ''' Writes fields to binary file'''
        fields = self.unflatten(U)
        return self.solver.write_fields(fields, idx, path)

    # --- AD methods (require JAX backend) ------------------------------------

    def ad_evolve(self, fields, T):
        """AD-compatible evolution on field lists. Requires JAX backend."""
        return self.solver.ad_evolve(fields, T)

    def jvp_evolve(self, fields, tangent_fields, T):
        """Forward-mode AD: exact tangent map J·v on field lists.

        Parameters
        ----------
        fields : list of arrays
            Base point (physical space).
        tangent_fields : list of arrays
            Tangent vector v (same structure as fields).
        T : float
            Integration time.

        Returns
        -------
        primals : list of arrays
            Evolved fields Φ(fields, T).
        tangents : list of arrays
            Tangent output J·v where J = dΦ/d(fields).
        """
        from .._backend import jvp
        primals, tangents = jvp(
            lambda f: self.solver.ad_evolve(f, T),
            (fields,), (tangent_fields,)
        )
        return primals, tangents

    def vjp_evolve(self, fields, cotangent_fields, T):
        """Reverse-mode AD: adjoint action J^T·v on field lists.

        Parameters
        ----------
        fields : list of arrays
            Base point (physical space).
        cotangent_fields : list of arrays
            Cotangent vector v (same structure as output fields).
        T : float
            Integration time.

        Returns
        -------
        primals : list of arrays
            Evolved fields Φ(fields, T).
        cotangents : list of arrays
            Adjoint output J^T·v where J = dΦ/d(fields).
        """
        from .._backend import vjp
        primals, vjp_fn = vjp(
            lambda f: self.solver.ad_evolve(f, T),
            fields
        )
        return primals, vjp_fn(cotangent_fields)

    # -------------------------------------------------------------------------

    def floquet_multipliers(self, fields, T, n, tol, ep0=1e-7, sx=None, b='U', test=False):
        ''' Calculates Floquet multipliers using the Arnoldi algorithm.
        Method follows Viswanath (2007), same as Newton method.

        Paramters
        ---------
        fields: list of arrays
            List of fields to be used.
        T: float
            Time to evolve in each arnoldi iteration.
        n: int
            Number of multiplier to calculate
        tol: float
            Tolerance of the Arnoldi iteration
        ep0: float, optional
            Perturbation factor. Default is 1e-7.
        sx: float or None, optional
            Translation in x direction. Default is None.
        b: str or np.array, optional
            Initial guess for Arnoldi. Default is 'U'. Could also take in
            'random' or a np.array of the same size as U

        Returns
        -------
        eigval_H: np.array of complex
            Floquet multipliers
        eigvec_H: np.arrays of complex
            Floquet vectors
        Q: np.arrays
            Arnoldi basis
        '''

        U = self.flatten(fields)
        UT = self.evolve(U, T)

        # Translate UT by sx
        if sx is not None:
            UT = self.translate(UT, sx)

        def apply_J(dU):
            ''' Applies J (jacobian of poincare map) matrix to vector dU  '''
            # 1e-7 factor chosen to balance accuracy and numerical stability
            epsilon = ep0*np.linalg.norm(U)/np.linalg.norm(dU)

            # Perturb U by epsilon*dU
            U_pert = U + epsilon*dU

            # Calculate derivative w.r.t. initial fields
            dUT_dU = self.evolve(U_pert, T)

            if sx is not None:
                dUT_dU = self.translate(dUT_dU, sx)

            dUT_dU = (dUT_dU - UT)/epsilon
            return dUT_dU

        if isinstance(b, str):
            if b == 'U':
                b = U
            elif b == 'random':
                b = np.random.randn(len(U))
            elif b == 'phases':
                b = self.phase_shifted_b(fields)
        elif isinstance(b, np.ndarray):
            pass
        else:
            raise ValueError('b must be one of the given options')

        eigval_H, eigvec_H, Q = arnoldi_eig(apply_J, b, n, tol)

        if test:
            # Checks if returned eigenvectors satisfy Av=lambda v
            Aev_r = np.zeros((len(U), n))
            Aev_i = np.zeros((len(U), n))
            for i in range(n):
                ev = Q @ eigvec_H[:,i]
                Aev_r[:,i] = apply_J(ev.real)
                Aev_i[:,i] = apply_J(ev.imag)

            return eigval_H, eigvec_H, Q, Aev_r, Aev_i

        return eigval_H, eigvec_H, Q

    def lyapunov_exponents(self, fields, T, n, nsteps, tol=1e-10, ep0=1e-7, sx=None, b='random', return_hist=False,
                           checkpoint_every=None, checkpoint_path='lyap_checkpoint.npz', restart=False):
        ''' Computes Lyapunov exponents and Kaplan–Yorke dimension via QR iteration.

        This method implements the Benettin algorithm for estimating finite-time
        Lyapunov exponents by repeatedly evolving a set of orthonormal perturbations
        over intervals of duration T and reorthonormalizing them using QR decomposition.

        Parameters
        ----------
        fields : list of fields
            Initial fields defining the state U.
        T : float
            Integration time between QR reorthonormalizations.
        n : int
            Number of Lyapunov exponents to compute.
        nsteps : int
            Number of reorthonormalization intervals (total time = nsteps * T)
        tol : float, optional
            QR tolerance. Default is 1e-10.
        ep0 : float, optional
            Perturbation scaling for the finite-difference tangent map.
            Default is 1e-7.
        sx : float, optional
            Translation in x direction (for translationally invariant systems).
        b : str or np.ndarray, optional
            Initial perturbation seed ('random', 'U', 'phases', or user-defined array).
            Ignored if restart=True.
        return_hist : bool, optional
            If True, also returns the history of Lyapunov exponents at each step.
        checkpoint_every : int or None, optional
            If set, saves checkpoint (U, Q, le_sum, le_hist, step) every this many steps.
        checkpoint_path : str, optional
            Path to checkpoint file. Default is 'lyap_checkpoint.npz'.
        restart : bool, optional
            If True, loads state from checkpoint_path and resumes from there.

        Returns
        -------
        lyap_exponents : np.ndarray, shape (n,)
            Sorted Lyapunov exponents in descending order.
        D_KY : float
            Kaplan–Yorke dimension computed from the cumulative sum of exponents.
        le_hist : np.ndarray, shape (nsteps, n), optional
            History of Lyapunov exponents at each reorthonormalization step.
        '''

        U = self.flatten(fields)

        def apply_J(U, dU, Uevol):
            ''' Applies the finite-time tangent map DΦ_T(U) to perturbation dU. '''
            epsilon = ep0 * np.linalg.norm(U) / np.linalg.norm(dU)
            U_pert = U + epsilon * dU
            dUT_dU = self.evolve(U_pert, T) - Uevol
            if sx is not None:
                dUT_dU = self.translate(dUT_dU, sx)
            return dUT_dU / epsilon

        # --- Initialize or restore state ---
        if restart:
            ckpt = np.load(checkpoint_path)
            U = ckpt['U']
            Q = ckpt['Q']
            le_sum = ckpt['le_sum']
            le_hist = ckpt['le_hist']
            start_step = int(ckpt['step']) + 1
            print(f"Restarting from checkpoint '{checkpoint_path}' at step {start_step}/{nsteps}")
        else:
            if isinstance(b, str):
                if b == 'U':
                    b = U.copy()
                elif b == 'random':
                    b = np.random.randn(len(U))
                elif b == 'phases':
                    b = self.phase_shifted_b(fields)
            elif not isinstance(b, np.ndarray):
                raise ValueError("b must be 'U', 'random', 'phases', or a NumPy array.")

            Q = np.zeros((len(U), n))
            Q[:, 0] = b / np.linalg.norm(b)
            for i in range(1, n):
                q = np.random.randn(len(U))
                for j in range(i):
                    q -= np.dot(Q[:, j], q) * Q[:, j]
                Q[:, i] = q / np.linalg.norm(q)

            le_sum = np.zeros(n)
            le_hist = np.zeros((nsteps, n))
            start_step = 0

        # --- Accumulate finite-time Lyapunov exponents ---
        for step in range(start_step, nsteps):
            # Propagate basis vectors through tangent map
            V = np.zeros_like(Q)
            Uevol = self.evolve(U, T)
            for i in range(n):
                V[:, i] = apply_J(U, Q[:, i], Uevol)

            # QR reorthonormalization
            Q, R = np.linalg.qr(V)
            diagR = np.abs(np.diag(R))
            le_sum += np.log(diagR + 1e-300)  # prevent log(0)

            # Store running average of Lyapunov exponents
            t = (step + 1) * T
            le_running = le_sum / t
            le_running = np.sort(le_running)[::-1]  # sort in descending order
            le_hist[step, :] = le_running

            if (step + 1) % 10 == 0:
                print(f"step {step+1}/{nsteps}  le = {np.array2string(le_running, precision=6, separator=', ')}")

            # Re-normalize columns to avoid drift
            for i in range(n):
                Q[:, i] /= np.linalg.norm(Q[:, i])

            U = np.copy(Uevol)

            if checkpoint_every is not None and (step + 1) % checkpoint_every == 0:
                np.savez(checkpoint_path, U=U, Q=Q, le_sum=le_sum, le_hist=le_hist, step=step)
                print(f"  checkpoint saved to '{checkpoint_path}'")

        # --- Average over total time ---
        lyap_exponents = le_sum / (nsteps * T)
        lyap_exponents = np.sort(lyap_exponents)[::-1]

        # --- Kaplan–Yorke dimension ---
        S = np.cumsum(lyap_exponents)
        positive = np.where(S >= 0)[0]
        if len(positive) > 0:
            j = positive[-1]
            if j + 1 < len(S):
                D_KY = j + S[j] / abs(lyap_exponents[j + 1])
            else:
                D_KY = float(j)
        else:
            D_KY = 0.0

        if not return_hist:
            return lyap_exponents, D_KY

        return lyap_exponents, D_KY, le_hist

class UPONewtonSolver(DynSys):
    """
    Implements the Newton-Krylov method for computing steady states and
    periodic orbits in PDE systems (e.g. Kolmogorov flow, Boussinesq
                                    equations).

    References:
      - Chandler & Kerswell (2013)
      - Viswanath (2007)
    """
    def __init__(self, solver, T, sx, sp, lda):
        super().__init__(self, solver)

    def form_X(self, U, T=None, sx=None, lda = None):
        ''' Form X vector from fields U, T, sx and lda (if applicable) '''
        X = np.copy(U)
        if self.pm.T is not None:
            X = np.append(X, T)
        if self.pm.sx is not None:
            X = np.append(X, sx)
        if lda is not None:
            X = np.append(X, lda)
        return X

    def unpack_X(self, X):
        '''X could contain extra params sx and lda if searching for RPOs (pm.sx
                                                                          != 0)
        or using arclength continuation'''

        # Determine size of U
        if getattr(self.pm, 'remove_boundary', False):
            dim_U = self.grid.Nx * (self.grid.Nz-2) * self.solver.num_fields
        else:
            dim_U = self.grid.N * self.solver.num_fields

        # Extract U, T, sx, lda from X
        U = X[:dim_U]
        idx = dim_U

        if self.pm.T is not None:
            T = X[idx] # T saved as first argument after U
            idx += 1
        else:
            T = self.pm.Tconst # if searching for TW or equilibrium T must be fixed at a small but not too small value

        sx = X[idx] if (self.pm.sx is not None) else 0.

        if not self.pm.arclength:
            return U, T, sx
        else:
            lda = X[-1]
            return U, T, sx, lda

    def get_restart_values(self, restart, iA: int | None = None):
        """Get values from last Newton iteration of T and sx from newton.txt"""
        path = self._get_path(self.pm.newton_dir, "newton.txt", iA=iA)

        # Load data and find restart iteration
        iters, T, sx = np.loadtxt(path, delimiter=',', skiprows=1, unpack=True, usecols=(0, 2, 3))
        idx_restart = np.argwhere(iters == restart)[0][0]

        values = [T[idx_restart], sx[idx_restart]]
        if self.pm.arclength:
            lda = np.loadtxt(path, delimiter=',', skiprows=1, unpack=True, usecols=5)
            values.append(lda[idx_restart])
        return values

    def form_b(self, U, UT):
        "Form RHS of extended Newton system. UT is evolved and translated flattened field"
        b = U - UT
        if self.pm.sx is not None:
            b = np.append(b, 0.)
        if self.pm.T is not None:
            b = np.append(b, 0.)
        return b

    def norm(self, U):
        return np.linalg.norm(U)

    def apply_proj(self, U, dU, sp):
        '''Applies projection if sp is True. To dU or U+dU'''
        if not sp:
            return U+dU

        if self.pm.sp_dU:
            dU = self.sol_project(dU)
            return U+dU
        else:
            return self.sol_project(U+dU)

    def update_A(self, X, iN):
        '''Creates (extended) Jacobian matrix to be applied to U throughout GMRes'''
        # Compute variables and derivatives used throughout gmres iterations
        U, T, sx = self.unpack_X(X)

        # Evolve fields and save output
        save_out = True if self.pm.save_outputs == 'all' else False
        save_bal = True if self.pm.save_balance == 'all' else False
        UT = self.evolve(U, T, save_out, save_bal, iN=iN-1)

        # Translate UT by sx and calculate derivatives
        if self.pm.sx is not None:
            UT = self.translate(UT, sx)

            dUT_ds = self.deriv_U(UT, self.grid.kx)
            dU_ds = self.deriv_U(U, self.grid.kx)
        else:
            dUT_ds = dU_ds = np.zeros_like(U) # No translation if sx is None

        # Calculate derivatives in time
        if self.pm.T is not None:
            dUT_dT = self.evolve(UT, self.solver.pm.dt)
            dUT_dT = (dUT_dT - UT)/self.solver.pm.dt

            dU_dt = self.evolve(U, self.solver.pm.dt)
            dU_dt = (dU_dt - U)/self.solver.pm.dt
        else:
            dUT_dT = dU_dt = np.zeros_like(U) # No evol if T is None


        def apply_A(dX):
            ''' Applies A (extended Jacobian) to vector X^t  '''
            dU, dT, ds = self.unpack_X(dX)

            print(self.norm(U), type(self.pm.eps0), self.norm(dU))
            epsilon = self.pm.eps0*self.norm(U)/self.norm(dU)

            # Perturb U by epsilon*dU and apply solenoidal projection if sp1 = True
            U_pert = self.apply_proj(U, epsilon*dU, self.pm.sp1)

            # Calculate derivative w.r.t. initial fields
            dUT_dU = self.evolve(U_pert, T)
            if self.pm.sx is not None:
                dUT_dU = self.translate(dUT_dU,sx)
            dUT_dU = (dUT_dU - UT)/epsilon

            # Calculate projections of dU needed for extended Newton system
            Tx_proj = np.dot(dU_ds.conj(), dU).real
            t_proj = np.dot(dU_dt.conj(), dU).real

            # Save norms for diagnostics
            norms = [self.norm(U_) for U_ in [U, dU, dUT_dU, dU_dt, dUT_dT, dU_ds, dUT_ds]]

            self.write_apply_A(iN, norms, t_proj, Tx_proj)

            # LHS of extended Newton system
            LHS = dUT_dU - dU + dUT_ds*ds + dUT_dT*dT
            if self.pm.T is not None:
                LHS = np.append(LHS, t_proj)
            if self.pm.sx is not None:
                LHS = np.append(LHS, Tx_proj)

            return LHS

        return apply_A, UT

    def run_newton(self, X):
        '''Iterates Newton-GMRes solver until convergence'''
        pm = self.pm
        if self.pm.restart_iN == 0:
            self.write_header_newton()

        for iN in range(pm.restart_iN+1, pm.N_newt):
            # Unpack X
            U, T, sx = self.unpack_X(X)

            # Calculate A matrix for newton iteration
            apply_A, UT = self.update_A(X, iN)

            # RHS of Newton extended system
            b = self.form_b(U, UT)
            F = self.norm(b) #||b|| = ||F||: rootsearch function

            # Write to txts
            self.write_newton(iN, F, U, sx, T)
            self.write_headers(iN)

            # Perform GMRes iteration
            # Returns H, beta, Q such that X = Q@y, y = H^(-1)@beta
            H, beta, Q = GMRES(apply_A, b, pm.N_gmres, pm.tol_gmres, pm.gmres_dir, iN)

            # Perform hookstep to adjust solution to trust region
            X, F_new, UT = self.hookstep(X, H, beta, Q, iN)

            # Update solution
            U, T, sx = self.unpack_X(X)

            # Select different initial condition from orbit if solution is not converging
            if ((F-F_new)/F) < pm.tol_nudge:
                U = self.evolve(U, T*self.pm.frac_nudge)

            # Termination condition
            if (F_new < pm.tol_newt) and ((F-F_new)/F < pm.tol_improve):

                # Final evolve to save outputs and balance
                save_out = True if self.pm.save_outputs in ('all', 'last') else False
                save_bal = True if self.pm.save_balance in ('all', 'last') else False
                UT = self.evolve(U, T, save_out, save_bal, iN=iN)

                if pm.sx is not None:
                    UT = self.translate(UT, sx)
                b = self.form_b(U, UT)
                F = self.norm(b) #||b|| = ||F||: rootsearch function
                # Write to txts
                self.write_newton(iN+1, F, U, sx, T)
                break


    def hookstep(self, X, H, beta, Q, iN, arclength:dict|None = None):
        ''' Performs hookstep on solution given by GMRes untill new |F| is less than previous |F| (or max iter of hookstep is reached) '''
        ''' If performing arclength continuation arclength contains a dict with the relevant quantities '''
        # Unpack X
        if not arclength:
            U, T, sx = self.unpack_X(X)
        else:
            U, T, sx, lda = self.unpack_X(X, arclength = True)

        #Initial solution from GMRes in basis Q
        y = backsub(H, beta)
        #Initial trust region radius
        Delta = self.norm(y)

        #Define trust_region function
        if not arclength:
            trust_region = self.trust_region_function(H, beta, iN, y)
        else:
            trust_region = self.trust_region_function(H, beta, iN, y, arclength["iA"])

        mu = 0.
        #Perform hookstep
        for iH in range(self.pm.N_hook):
            y, mu = trust_region(Delta, mu)
            dx = Q@y #Unitary transform back to full dimension

            if not arclength:
                dU, dT, dsx = self.unpack_X(dx)
            else:
                dU, dT, dsx, dlda = self.unpack_X(dx, arclength = True)

            U_new = self.apply_proj(U, dU, self.pm.sp2)

            if self.pm.sx is not None:
                sx_new = sx+dsx.real
            else:
                sx_new = 0.

            if self.pm.T:
                T_new = T+dT.real
            else:
                T_new = self.pm.Tconst

            X_new = self.form_X(U_new, T_new, sx_new)

            if arclength:
                lda_new = lda+dlda.real
                X_new = np.append(X_new, lda_new)
                self.update_lda(lda_new)

            UT = self.evolve(U_new, T_new)
            if self.pm.sx is not None:
                UT = self.translate(UT,sx_new)

            b = self.form_b(U_new, UT)
            if arclength:
                # add arclength term
                N = self.N_constraint(X_new, arclength["dX_dr"], arclength["dr"], arclength["X1"])
                b = np.append(b, -N)

            F_new = self.norm(b)
            lin_exp = self.norm(beta - self.pm.c * H @ y) #linear expansion of F around x (in basis Q).
            # beta = H@y holds

            if arclength:
                self.write_hookstep(iN, iH, F_new, lin_exp, arclength["iA"])
            else:
                self.write_hookstep(iN, iH, F_new, lin_exp)

            if F_new <= lin_exp:
                break
            else:
                Delta *= self.pm.reduc_reg #reduce trust region
        return X_new, F_new, UT


    def trust_region_function(self, H, beta, iN, y0, iA:int|None =None):
        ''' Performs trust region on solution provided by GMRes. Must be instantiated at each Newton iteration '''
        R = H #R part of QR decomposition
        A_ = R.T @ R #A matrix in trust region
        b = R.T @ beta #b vector in trust region

        def trust_region(Delta, mu):
            ''' Delta: trust region radius, mu: penalty parameter  '''
            if mu == 0: #First hoostep iteration. Doens't perform trust region
                y_norm = self.norm(y0)
                Delta0 = y_norm #Initial trust region

                self.write_trust_region(iN, Delta0, mu, y0, A_, iA)

                mu = self.pm.mu0 #Initialize first nonzero value of mu
                return y0, mu

            for _ in range(1, 1000): #1000 big enough to ensure that condition is satisfied

                # Ridge regression adjustment
                A = A_ + mu * np.eye(A_.shape[0])
                y = np.linalg.solve(A, b) #updates solution with trust region

                self.write_trust_region(iN, Delta, mu, y, A, iA)

                if self.norm(y) <= Delta:
                    break
                else:
                    # Increase mu until trust region is satisfied
                    mu *= self.pm.mu_inc

            return y, mu
        return trust_region

    def mkdir(self, path):
        os.makedirs(path, exist_ok=True)

    def _get_path(self, base_dir: str | None, *parts, iA: int | None = None):
        """Safely join base_dir with optional suffix and subpaths.
        Adds iA (arclength iteration) suffix if provided."""
        if base_dir is None:
            return None
        if iA is not None:
            base_dir = f"{base_dir}{iA:02}"
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, *parts)

    def write_header_newton(self, iA: int|None = None):
        """Writes header to newton.txt file."""
        path = self._get_path(self.pm.newton_dir, "newton.txt", iA=iA)
        if path is None:
            return

        header = "iN, |F|, T, sx, |U|"
        if self.pm.arclength:
            header += ", lambda, N(X)"
        print(header, file=open(path, "a"))

    def write_newton(self, iN, F, U, sx, T, arclength: tuple|None = None, iA: int|None = None):
        """Writes iteration data to print file."""
        newton_path = self._get_path(self.pm.newton_dir, "newton.txt", iA=iA)
        if newton_path is None:
            return

        content = f"{iN-1:02},{F:.6e},{T},{sx:.8e},{self.norm(U):.6e}"
        if arclength:
            content += f",{arclength[0]},{arclength[1]}"
        print(content, file=open(newton_path, "a"))

    def write_headers(self, iN, iA: int | None = None):
        """Writes headers to diagnostic files for every Newton iteration iN."""
        suffix = f'iN{iN:02}.txt'
        gmres_path = self._get_path(self.pm.gmres_dir, 'gmres_'+suffix, iA=iA)
        apply_A_path = self._get_path(self.pm.apply_A_dir, 'apply_A_'+suffix, iA=iA)
        hook_path = self._get_path(self.pm.hookstep_dir, 'hookstep_'+suffix, iA=iA)
        trust_region_path = self._get_path(self.pm.trust_region_dir, 'trust_region_'+suffix, iA=iA)

        if gmres_path:
            print('iG, error', file=open(gmres_path, "w"))

        if hook_path:
            print('iH, |F|, |F(x)+cAdx|, |F(x)+Adx|', file=open(hook_path, "w"))

        if apply_A_path:
            lda = (", |dUT/dlda|", ", lda_proj") if self.pm.arclength else ("", "")
            header = f"|U|, |dU|, |dUT/dU|, |dU/dt|, |dUT/dT|, |dU/ds|, |dUT/ds|{lda[0]}, t_proj, Tx_proj{lda[1]}"
            print(header, file=open(apply_A_path, "w"))

        if trust_region_path:
            print("Delta, mu, |y|, cond(A)", file=open(trust_region_path, "w"))

    def write_apply_A(self, iN, norms, t_proj, Tx_proj, lda_proj=None, iA: int | None = None):
        path = self._get_path(self.pm.apply_A_dir, f"apply_A_iN{iN:02}.txt", iA=iA)
        if path is None:
            return

        content = ",".join([f"{norm:.4e}" for norm in norms]) + f",{t_proj:.4e},{Tx_proj:.4e}"
        if lda_proj:
            content += f",{lda_proj:.4e}"
        print(content, file=open(path, "a"))

    def write_hookstep(self, iN, iH, F_new, lin_exp, iA: int | None = None):
        path = self._get_path(self.pm.hookstep_dir, f"hookstep_iN{iN:02}.txt", iA=iA)
        if path is None:
            return

        print(f"{iH:02},{F_new:.4e},{lin_exp:.4e}", file=open(path, "a"))

    def write_trust_region(self, iN, Delta, mu, y, A, iA: int | None = None):
        path = self._get_path(self.pm.trust_region_dir, f"trust_region_iN{iN:02}.txt", iA=iA)
        if path is None:
            return

        print(f"{Delta:.4e},{mu:.4e},{self.norm(y):.4e},{np.linalg.cond(A):.3e}", file=open(path, "a"))

    def phase_shifted_b(self, fields):
        """Generate b vector by modifying phases in Fourier space."""
        def apply_phase_shift(U):
            """Applies a random phase shift to a single field U and projects to solenoidal modes."""
            U_hat = self.grid.forward(U)
            random_phases = np.exp(1j * np.random.uniform(0, 2*np.pi, U_hat.shape))
            U_hat_shifted = np.abs(U_hat) * random_phases  # Preserve magnitudes
            return U_hat_shifted

        # Separate into u and v components
        u, v = fields
        b_vector = []

        u_hat_shifted = apply_phase_shift(u)
        v_hat_shifted = apply_phase_shift(v)

        # Project in Fourier space
        u_proj_hat, v_proj_hat = self.grid.inc_proj((u_hat_shifted, v_hat_shifted))

        # Inverse FFT to physical space
        u_proj = self.grid.inverse(u_proj_hat)
        v_proj = self.grid.inverse(v_proj_hat)
        fields_proj = [u_proj, v_proj]

        b_vector = self.flatten([f.flatten() for f in fields_proj])
        return b_vector


    def run_arclength(self, X0, X1, X_restart = None, iA:int|None = None):
        '''Iterates Newton-GMRes solver until convergence using arclength continuation
        Follows convention of Chandler - Kerswell: Invariant recurrent solutions..
        X: Vector containing (U, T, sx, lda) to be updated with Newton until it converges to periodic orbit
        dX_dr: Derivative of X w.r.t. arclength
        X1: Previous converged solution (X(r0) in Chandler-Kerswell
        iA: arclength iteration in case automatic arclength is performed '''

        # Define relevant quantities for arclength continuation
        dr = np.linalg.norm(X1 - X0)
        dX_dr = (X1 - X0)/dr

        if X_restart is not None: # in case Newton iteration is to be restarted
            X = X_restart
        else:
            X = X1

        start_iN = self.pm.restart_iN+1 if (iA==self.pm.restart_iA) else 1

        for iN in range(start_iN, self.pm.N_newt):
            # lda: lambda parameter, Re (Reynolds) if solver=='KolmogorovFlow', Ra (Rayleigh) if solver=='BOUSS'
            U, T, sx, lda = self.unpack_X(X, arclength = True)

            # Calculate A matrix for newton iteration
            apply_A, UT = self.update_A_arc(X, dX_dr, iN, iA)

            # RHS of Newton extended system
            # form b (rhs)
            b = self.form_b(U, UT)

            # add arclength term
            N = self.N_constraint(X, dX_dr, dr, X1)
            b = np.append(b, -N)

            F = self.norm(b[:-1]) # F only includes diff between U and UT, not arclength constraint

            # Write to txts
            self.write_newton(iN, F, U, sx, T, (lda*self.pm.norm, N), iA)
            self.write_headers(iN, iA)

            # Perform GMRes iteration
            # Returns H, beta, Q such that X = Q@y, y = H^(-1)@beta
            H, beta, Q = GMRES(apply_A, b, self.pm.N_gmres, self.pm.tol_gmres, self.pm.gmres_dir, iN, iA)

            # Perform hookstep to adjust solution to trust region
            X, F_new, UT = self.hookstep(X, H, beta, Q, iN, arclength = {'iA':iA, 'dX_dr':dX_dr, 'dr':dr, 'X1':X1})

            # Update solution
            U, T, sx, lda = self.unpack_X(X, arclength = True)

            # Termination condition
            if (F_new < self.pm.tol_newt) and ((F-F_new)/F < self.pm.tol_improve):
                UT = self.evolve(U, T, save = True, iN = iN, iA = iA)
                if self.pm.sx is not None:
                    UT = self.translate(UT, sx)
                b = self.form_b(U, UT)
                # add arclength term
                N = self.N_constraint(X, dX_dr, dr, X1)
                b = np.append(b, -N)
                F = self.norm(b[:-1]) # F only includes diff between U and UT, not arclength constraint
                # Write to txts
                self.write_newton(iN+1, F, U, sx, T, (lda*self.pm.norm, N), iA)

                if iA is not None:
                    self.save_converged(X, iA)
                return X

    def save_converged(self, X, iA):
        """Save converged solution fields and metadata."""
        U, T, sx, lda = self.unpack_X(X, arclength=True)

        # Use parameter-driven converged directory
        path = self._get_path(self.pm.converged_dir, "solver.txt", iA=iA)
        os.makedirs(path, exist_ok=True)

        solver_file = os.path.join(path, "solver.txt")
        if iA == 0 and not os.path.exists(solver_file):
            print("iA, T, sx, lda, |U|", file=open(solver_file, "a"))

        print(f"{iA:02},{T},{sx:.8e},{lda*self.pm.norm},{self.norm(U):.6e}",
            file=open(solver_file, "a"))

        self.write_fields(U, idx=0, path=path)

    def update_lda(self, lda):
        '''Updates lambda parameter in solver. norm: normalization parameter'''
        if self.solver.solver == 'KolmogorovFlow':
            Re = lda # Reynolds number
            self.pm.nu = 1 / Re
        elif self.solver.solver == 'BOUSS':
            Ra = lda # Rayleigh number
            Ra *= self.pm.norm # to account for possible normalization
            self.pm.ra = Ra
        else:
            raise Exception('Arclength only implemented for Kolmogorov and BOUSS')

    def N_constraint(self, X, dX_dr, dr, X1):
        '''Calculates N function resulting from the arclength constraint as in
        'Chandler - Kerswell: Invariant recurrent solutions..' '''
        alpha = getattr(self.pm, 'alpha', 1.) # parametrization velocity. default = 1.
        return np.dot(dX_dr, (X-X1)) - dr * alpha**2

    def update_A_arc(self, X, dX_dr, iN, iA: int|None):
        '''Creates (extended) Jacobian matrix to be applied to U throughout GMRes'''
        # Compute variables and derivatives used throughout gmres iterations
        U, T, sx, lda = self.unpack_X(X, arclength = True)

        # Update lambda
        self.update_lda(lda)

        # Evolve fields and save output
        UT = self.evolve(U, T, save = True, iN = iN-1, iA = iA)

        # Translate UT by sx and calculate derivatives
        if self.pm.sx is not None:
            UT = self.translate(UT, sx)

            dUT_ds = self.deriv_U(UT, self.grid.kx)
            dU_ds = self.deriv_U(U, self.grid.kx)
        else:
            dUT_ds = dU_ds = np.zeros_like(U) # No translation if sx is None

        # Calculate derivatives in time
        if self.pm.T is not None:
            dUT_dT = self.evolve(UT, self.solver.pm.dt)
            dUT_dT = (dUT_dT - UT)/self.solver.pm.dt

            dU_dt = self.evolve(U, self.solver.pm.dt)
            dU_dt = (dU_dt - U)/self.solver.pm.dt
        else:
            dUT_dT = dU_dt = np.zeros_like(U) # No evol if T is None

        # Calculate derivative of evolved translated fields wrt lambda
        dlda = lda * 1e-3
        self.update_lda(lda + dlda)
        dUT_dlda = self.evolve(U, T)
        dUT_dlda = self.translate(dUT_dlda,sx)
        dUT_dlda = (dUT_dlda - UT)/dlda

        # Return lambda to previous state
        self.update_lda(lda)

        def apply_A(dX):
            ''' Applies A (extended Jacobian) to vector X^t  '''
            dU, dT, ds, dlda = self.unpack_X(dX, arclength = True)

            # 1e-7 factor chosen to balance accuracy and numerical stability
            epsilon = 1e-7*self.norm(U)/self.norm(dU)

            # Perturb U by epsilon*dU and apply solenoidal projection if sp1 = True
            U_pert = self.apply_proj(U, epsilon*dU, self.pm.sp1)

            # Calculate derivative w.r.t. initial fields
            dUT_dU = self.evolve(U_pert, T)
            if self.pm.sx is not None:
                dUT_dU = self.translate(dUT_dU,sx)
            dUT_dU = (dUT_dU - UT)/epsilon

            # Calculate projections of dU needed for extended Newton system
            Tx_proj = np.dot(dU_ds.conj(), dU).real
            t_proj = np.dot(dU_dt.conj(), dU).real
            lda_proj = np.dot(dX_dr.conj(), dX).real

            # Save norms for diagnostics
            norms = [self.norm(U_) for U_ in [U, dU, dUT_dU, dU_dt, dUT_dT, dU_ds, dUT_ds, dUT_dlda]]

            self.write_apply_A(iN, norms, t_proj, Tx_proj, lda_proj, iA)

            # LHS of extended Newton system
            LHS = dUT_dU - dU + dUT_ds*ds + dUT_dT*dT + dUT_dlda*dlda
            if self.pm.T is not None:
                LHS = np.append(LHS, t_proj)
            if self.pm.sx is not None:
                LHS = np.append(LHS, Tx_proj)
            # Add arclength term
            LHS = np.append(LHS, lda_proj)

            return LHS

        return apply_A, UT

    def run_arc_auto(self, X0, X1, X_restart = None):
        '''Iterates Newton-GMRes solver until convergence using arclength continuation
        Once a solution has converged it automatically uses the new converged solution as
        the new starting point
        '''
        start_iA = max(2, self.pm.restart_iA)

        for iA in range(start_iA, self.pm.N_arc):
            if (self.pm.restart_iN == 0) or (iA != start_iA):
                self.write_header_newton(iA = iA)

            X = self.run_arclength(X0, X1, X_restart, iA)
            X0 = X1
            X1 = X
            X_restart = None
