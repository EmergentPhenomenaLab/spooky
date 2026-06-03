"""Gradient-based UPO initializer via periodicity-loss minimization.

Reference: Page, Norgaard, Brenner & Kerswell, PNAS 2024.
arXiv:2212.11886
"""
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from .dyn_sys import DynSys
from .._backend import JAX_ENABLED


class UPOGradientDescent(DynSys):
    """Find UPO initial guesses by minimizing the periodicity loss with L-BFGS-B.

    Minimizes L(u0, T) = ||Φ(u0, T) - u0||² / ||u0||² using exact JAX
    gradients, generating high-quality seeds for Newton-Krylov-Hookstep.

    T is treated as a fixed parameter (integer multiple of dt). To search
    over periods, run with several values of T in an outer loop.

    Parameters
    ----------
    solver : PseudoSpectral instance
        The PDE solver to wrap.

    Example
    -------
    >>> gd = UPOGradientDescent(solver)
    >>> fields_guess, loss = gd.run(fields0, T=10.0, n_iter=500)
    >>> # Hand off to Newton-Krylov:
    >>> U_guess = ds.flatten(fields_guess)
    """

    def __init__(self, solver):
        super().__init__(solver)
        if not JAX_ENABLED:
            raise RuntimeError(
                "UPOGradientDescent requires JAX backend (set NUMPY_BACKEND=jax)."
            )

    def loss(self, fields, T):
        """Normalized periodicity residual ||Φ(u,T) - u||² / ||u||²."""
        import jax.numpy as jnp
        out = self.solver.ad_evolve(fields, T)
        res = [o - f for o, f in zip(out, fields)]
        num = sum(jnp.sum(r ** 2) for r in res)
        den = sum(jnp.sum(f ** 2) for f in fields)
        return num / den

    def run(self, fields0, T, n_iter=500, tol=1e-8, verbose=True):
        """Minimize periodicity loss from initial condition fields0 at period T.

        Parameters
        ----------
        fields0 : list of array_like
            Initial condition in physical space (any numeric arrays).
        T : float
            Period (fixed; must be an integer multiple of solver.grid.dt).
        n_iter : int
            Maximum L-BFGS-B iterations. Default 500.
        tol : float
            Gradient norm tolerance for convergence. Default 1e-8.
        verbose : bool
            Print initial/final loss and convergence message. Default True.

        Returns
        -------
        fields_opt : list of np.ndarray
            Optimized initial condition in physical space.
        loss_final : float
            Final periodicity loss value.
        """
        import jax
        import jax.numpy as jnp

        shapes = [np.array(f).shape for f in fields0]
        sizes  = [int(np.prod(s)) for s in shapes]

        def pack(fields):
            return np.concatenate(
                [np.array(f).ravel() for f in fields]
            ).astype(np.float64)

        def unpack(x):
            fields, idx = [], 0
            for s, sz in zip(shapes, sizes):
                fields.append(jnp.array(x[idx:idx + sz].reshape(s)))
                idx += sz
            return fields

        val_and_grad = jax.jit(jax.value_and_grad(lambda f: self.loss(f, T)))

        def fg(x):
            fields = unpack(x)
            val, g_fields = val_and_grad(fields)
            g = np.concatenate(
                [np.array(gi).ravel() for gi in g_fields]
            ).astype(np.float64)
            return float(val), g

        x0 = pack(fields0)

        if verbose:
            print(f"UPO gradient descent  T={T}  n_iter={n_iter}")
            print(f"  initial loss = {fg(x0)[0]:.4e}")

        result = scipy_minimize(
            fg, x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': n_iter, 'gtol': tol},
        )

        fields_opt = [np.array(f) for f in unpack(result.x)]

        if verbose:
            status = 'converged' if result.success else result.message
            print(f"  final loss   = {result.fun:.4e}  "
                  f"({result.nit} iters, {status})")

        return fields_opt, float(result.fun)
