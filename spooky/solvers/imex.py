''' Abstract IMEX (implicit-explicit) spectral solver class.

Sibling to :class:`PseudoSpectral` for wall-bounded problems where the stiff
terms (diffusion + the incompressibility/pressure constraint) are treated
implicitly and the cheap terms (advection, buoyancy) explicitly.

Unlike the fully-explicit ``rkstep`` interface, an IMEX step solves one or more
*linear* systems with state-independent matrices that depend on the timestep.
Rather than taking a fractional final step (which would need a second scheme),
``evolve``/``ad_evolve`` realise an arbitrary ``T`` exactly by taking
``Nt = round(T/grid.dt)`` uniform steps of ``dt = T/Nt``; the operator set for
that ``dt`` comes from :meth:`_ops_for` (subclasses cache a few of them).  The
map is therefore continuous in ``T`` (the Newton-Krylov solvers need that) and
each step remains an exact linear solve, which is what makes it differentiable:
JAX differentiates ``A^{-1} b`` exactly.  ``dt`` and the operators are passed to
the step functions as (traced) arguments, so one compiled step serves every ``T``.

Multistep schemes (e.g. SBDF2) are supported through a small state protocol:
``init_state`` builds the scheme's carry from the initial fields (taking
``n_startup`` startup steps to fill the extra time levels), ``step_state``
advances the carry by one ``dt``, and ``state_fields`` extracts the current
fields from it.  Single-step schemes use the identity defaults below.
'''
import abc

from .solver import Solver
from .._backend import xnp, JAX_ENABLED, lax_scan
from .. import pseudo as ps


class IMEXSpectral(Solver, abc.ABC):
    ''' Abstract IMEX spectral solver.

    Subclasses implement :meth:`imex_step`, which advances the mixed-space
    fields (Fourier in the periodic direction, physical/collocation in the
    wall-bounded direction) by one timestep ``dt``.  Multistep schemes
    additionally override :meth:`init_state` / :meth:`step_state` /
    :meth:`state_fields` and set ``n_startup``.
    '''

    #: number of steps consumed by init_state to fill the multistep carry
    n_startup = 0

    def __init__(self, grid: ps.Grid2D_wall):
        super().__init__(grid)

    def _nsteps(self, T):
        return int(round(T / self.grid.dt))

    def _steps_dt(self, T):
        ''' (Nt, dt) realising time T: Nt = round(T/grid.dt) steps of
        dt = T/Nt (within a factor 1 +- 1/(2 Nt) of grid.dt), with dt rounded
        to 13 significant digits so that nearly identical T's share one
        operator set instead of triggering a rebuild (the realised time
        deviates from T by < 1e-12 relative). '''
        Nt = self._nsteps(T)
        if Nt == 0:
            return 0, self.grid.dt
        return Nt, float(f'{T / Nt:.12e}')

    def _ops_for(self, dt):
        ''' State-independent implicit operators for timestep ``dt`` (a pytree
        of arrays, passed to the step functions as ``ops``).  Subclasses build
        and cache them; the default (no operators) is None. '''
        return None

    # -- multistep state protocol (identity defaults for single-step schemes) --

    def init_state(self, fields, dt, ops=None, **kw):
        ''' Build the scheme carry from the initial fields, advancing
        ``n_startup`` steps. Single-step default: the fields themselves. '''
        return fields

    def step_state(self, state, dt, ops=None, **kw):
        ''' Advance the scheme carry by one timestep. '''
        return self.imex_step(state, dt, ops=ops, **kw)

    def state_fields(self, state):
        ''' Extract the current fields from the scheme carry. '''
        return state

    def _step_kwargs(self):
        ''' Per-call keyword arguments for init_state/step_state (explicit-term
        parameters that must be *traced*, not baked into a compiled step).
        Parameters that enter the implicit operators belong in the
        ``_ops_for`` cache key instead. '''
        return {}

    def _compiled_steps(self):
        ''' (init, step) callables ``init(fields, dt, ops, **kw)`` and
        ``step(state, dt, ops, **kw)``.  Under JAX they are jitted once per
        solver with ``dt``, ``ops`` and the ``_step_kwargs`` entries as traced
        arguments (so no recompilation when T, hence dt, changes); under NumPy
        they are the plain methods. '''
        if not JAX_ENABLED:
            return (lambda f, dt, ops, **kw: self.init_state(f, dt, ops=ops, **kw),
                    lambda s, dt, ops, **kw: self.step_state(s, dt, ops=ops, **kw))
        if '_step_jit' not in self.__dict__:
            import jax
            self._step_jit = (
                jax.jit(lambda f, dt, ops, **kw: self.init_state(f, dt, ops=ops, **kw)),
                jax.jit(lambda s, dt, ops, **kw: self.step_state(s, dt, ops=ops, **kw)))
        return self._step_jit

    def _scan_steps(self, state, dt, nsteps, checkpoint_every=None, **kw):
        ''' Advance ``state`` by ``nsteps`` inside lax.scan.

        ``checkpoint_every`` (int or 'auto' = ~sqrt(nsteps)) wraps chunks of
        that many steps in ``jax.checkpoint``: reverse-mode memory drops from
        nsteps x (per-step tape) to about (nsteps/chunk) x carry + chunk x tape,
        at ~2x the forward cost.  None keeps the plain scan (O(nsteps) tape). '''
        # dt as an array constant (not a Python literal) so the traced scan
        # body is identical for every T and eager JVPs reuse their compilation
        dt = xnp.asarray(dt)
        step = lambda carry, _: (self.step_state(carry, dt, **kw), None)
        if nsteps <= 0:
            return state
        if checkpoint_every == 'auto':
            checkpoint_every = max(1, int(round(nsteps ** 0.5)))
        if not checkpoint_every or nsteps <= checkpoint_every:
            state, _ = lax_scan(step, state, None, length=nsteps)
            return state
        import jax
        n_in = int(checkpoint_every)
        n_out, rem = divmod(nsteps, n_in)

        def chunk(carry, _):
            carry, _ = lax_scan(step, carry, None, length=n_in)
            return carry, None

        state, _ = lax_scan(jax.checkpoint(chunk, prevent_cse=False),
                            state, None, length=n_out)
        if rem:
            state, _ = lax_scan(step, state, None, length=rem)
        return state

    # --------------------------------------------------------------------

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
        ''' Evolve fields to time T (exactly) using the IMEX step.

        ``T`` is realised with ``round(T/grid.dt)`` steps of ``T/round(T/grid.dt)``
        (see class docstring); the balance/output step labels use ``grid.dt``. '''
        if realfields:
            fields = [self.grid.forward(ff) for ff in fields]

        Nt, dt = self._steps_dt(T)

        if write_outputs:
            self.write_outputs(fields, 0, bstep, ostep, sstep,
                               bpath, opath, spath)

        if Nt > 0:
            kw = self._step_kwargs()
            ops = self._ops_for(dt)
            init, step_fn = self._compiled_steps()       # jitted per step under JAX
            state = init(fields, dt, ops, **kw)
            for step in range(self.n_startup, Nt):
                if step > 0 and write_outputs:
                    self.write_outputs(self.state_fields(state), step,
                                       bstep, ostep, sstep,
                                       bpath, opath, spath)
                state = step_fn(state, dt, ops, **kw)
            fields = self.state_fields(state)

        if write_outputs:
            self.write_outputs(fields, Nt, bstep, ostep, sstep,
                               bpath, opath, spath, final=True)

        if realfields:
            fields = [self.grid.inverse(ff) for ff in fields]
        return fields

    def ad_evolve(self, fields, T, checkpoint_every=None, **step_kwargs):
        ''' JAX-traceable evolution via ``lax.scan``. No file I/O.

        Differentiable end-to-end with jax.grad/jvp/vjp (requires JAX backend).
        Startup steps of a multistep scheme run outside the scan (they are a
        fixed, small number of ordinary traced steps).  ``checkpoint_every``
        enables rematerialisation for reverse mode (see ``_scan_steps``);
        extra keyword arguments are passed to init_state/step_state.
        '''
        if not JAX_ENABLED:
            raise NotImplementedError(
                "ad_evolve requires JAX backend (set NUMPY_BACKEND=jax)."
            )

        fields = [self.grid.forward(ff) for ff in fields]
        Nt, dt = self._steps_dt(T)

        if Nt > 0:
            ops = self._ops_for(dt)
            state = self.init_state(fields, dt, ops=ops, **step_kwargs)
            state = self._scan_steps(state, dt, Nt - self.n_startup,
                                     checkpoint_every, ops=ops, **step_kwargs)
            fields = self.state_fields(state)

        return [self.grid.inverse(ff) for ff in fields]

    @abc.abstractmethod
    def imex_step(self, fields, dt, ops=None, **kw):
        ''' Advance mixed-space fields by one timestep dt with the operator
        set ``ops`` (``self._ops_for(dt)`` when None). '''
        return []
