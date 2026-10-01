import numpy as np

# Box: periodic in x, walls in z. SPECTER convention (Lx = 2 pi, Lz = pi, aspect ratio 2);
# the solver works in free-fall units for any Lz.
Lx = 2 * np.pi
Lz = np.pi
Nx = 128
Nz = 65

# Physics
Ra = 1.0e5          # Rayleigh number (well above onset, Ra_c ~ 1708 no-slip)
Pr = 1.0            # Prandtl number
bc = 'no-slip'      # or 'free-slip'

# Time marching (free-fall units)
dt = 0.02
Tevolve = 200.0
bstep = 250         # balance (KE, Nusselt) output step
ostep = 2500        # field output step
