import abc

from .solver import Solver
from .._backend import xnp, copy_arr, JAX_ENABLED, lax_scan
from .. import pseudo as ps

class PseudoSpectral(Solver, abc.ABC):
    ''' Abstract Pseudospectral solver class

    All solvers must implement the following methods:
        - rkstep: Runge-Kutta step
        - balance: Energy balance
        - spectra: Energy spectra
        - outs: Outputs
    '''
    def __init__(self,
                 grid: ps.Grid1D | ps.Grid2D | ps.Grid2D_semi | ps.Grid3D,
                 rkord: int):
        super().__init__(grid)
        self.rkord = rkord

    def evolve(self,
               fields,
               T,
               bstep=None,
               ostep=None,
               sstep=None,
               bpath='',
               opath='',
               spath='',
               write_outputs=True,
               realfields=True):
        ''' Evolves velocity fields to time T '''

        # Forward transform
        if realfields:
            fields = [self.grid.forward(ff) for ff in fields]

        Nt = int(T/self.grid.dt)
        for step in range(Nt+1):
            # Store previous time step
            prev = copy_arr(fields)

            # Write outputs
            if write_outputs:
                self.write_outputs(fields, step, bstep, ostep, sstep, bpath, opath, spath)

            dt = self.grid.dt
            #Make exact last step
            if step == Nt:
                dt = T - Nt*self.grid.dt
                if xnp.isclose(dt, 0., atol=1e-8):
                    break

            # Time integration
            for oo in range(self.rkord, 0, -1):
                fields = self.rkstep(fields, prev, oo, dt)

        # Write final outputs
        if write_outputs:
            self.write_outputs(fields, Nt, bstep, ostep, sstep, bpath, opath, spath, final=True)

        # Inverse transform
        if realfields:
            fields = [self.grid.inverse(ff) for ff in fields]

        return fields

    def ad_evolve(self, fields, T):
        """JAX-traceable time evolution using lax.scan. No file I/O.

        Mirrors evolve() but is differentiable end-to-end with jax.grad/jvp/vjp.
        Requires JAX backend (NUMPY_BACKEND=jax).
        """
        if not JAX_ENABLED:
            raise NotImplementedError(
                "ad_evolve requires JAX backend (set NUMPY_BACKEND=jax)."
            )

        fields = [self.grid.forward(ff) for ff in fields]
        Nt = int(T / self.grid.dt)
        dt = self.grid.dt
        rkord = self.rkord

        def step_fn(carry, _):
            flds = carry
            prev = flds  # immutable JAX arrays; no copy needed
            for oo in range(rkord, 0, -1):
                flds = self.rkstep(flds, prev, oo, dt)
            return flds, None

        fields, _ = lax_scan(step_fn, fields, None, length=Nt)
        return [self.grid.inverse(ff) for ff in fields]

    @abc.abstractmethod
    def rkstep(self, fields, prev, oo, dt):
        return []
