''' 2D Rayleigh-Benard solver '''

import numpy as np
import os
import subprocess

from .solver import Solver
from .. import pseudo as ps

class SPECTER(Solver):
    '''
    SPECTER 2D flows solver. Solves Rayleigh-Benard if solver='BOUSS'
    '''

    num_fields = 3
    dim_fields = 2

    def __init__(self,
                 grid: ps.Grid2D_semi,
                 ra: float,
                 pr: float,
                 gamma: float = 1.,
                 solver: str ='BOUSS',
                 ftypes: list =['vx', 'vz', 'th'],
                 precision: str = 'double',
                 ext: int = 5,
                 nprocs: int = None):
        super().__init__(grid)
        self.grid = grid
        self.ra = ra
        self.pr = pr
        self.gamma = gamma
        self.solver = solver
        self.ftypes = ftypes
        self.precision = precision
        self.ext = ext

        if nprocs is not None:
            self.nprocs = nprocs
        else:
            self.nprocs = max(1, int(os.environ.get('SLURM_NTASKS', 2)) - 1)

    def evolve(self, fields, T, ipath=None, opath = '.', bpath='.', ostep=0, bstep=0):
        '''Evolves fields in T time and translates by sx. Calls Fortran'''
        if ipath is None:
            ipath = opath

        #write initial fields
        stat = 1
        self.write_fields(fields, path=ipath, idx=stat)

        #change parameters
        self.ch_params(T, ipath, opath, bstep, ostep, stat) #save fields every ostep, bal every bstep, spectrum every sstep

        #run SPECTER
        subprocess.run(f'mpirun -n {self.nprocs} ./{self.solver.upper()}', shell = True)

        #save balance prints
        if bstep:
            txts = 'balance.txt helicity.txt scalar.txt noslip_diagnostic.txt scalar_constant_diagnostic.txt'
            subprocess.run(f'mv {txts} {bpath}/.', shell = True)

        #load evolved fields
        if ostep==0:
            fields = self.load_fields(path=opath, idx = 2)
        else:
            fields = self.load_fields(path=opath, idx = int(T/self.grid.dt //ostep) + 1) # +1 since we start from 1
        return fields

    def save_binary_file(self, path, data):
        '''writes fortran file'''
        dtype = np.float64 if self.precision == 'double' else np.float32
        data = data.astype(dtype).reshape(data.size,order='F')
        data.tofile(path)

    def write_fields(self, fields, path, idx):
        ''' Writes fields to binary file.'''
        if not os.path.exists(path):
            os.makedirs(path)

        for field, ftype in zip(fields, self.ftypes):
            self.save_binary_file(os.path.join(path,f'{ftype}.{idx:0{self.ext}}.out'), field)

        # Save additional empty fields required by solver
        dtype = np.float64 if self.precision == 'double' else np.float32
        empty_field = np.zeros(self.grid.shape, dtype=dtype)
        for ftype in ('vy', 'pr'):
            self.save_binary_file(os.path.join(path,f'{ftype}.{idx:0{self.ext}}.out'), empty_field)

    def load_fields(self, path, idx): 
        '''Loads binary fields. '''
        dtype = np.float64 if self.precision == 'double' else np.float32

        fields = []
        for ftype in self.ftypes:
            file = os.path.join(path,f'{ftype}.{idx:0{self.ext}}.out')
            fields.append(np.fromfile(file,dtype=dtype).reshape(self.grid.shape,order='F'))
        return fields

    def get_nu_kappa(self):
        '''Calculates nu and kappa from ra'''
        Lz = self.grid.Lz

        nu = self.gamma*Lz**2 * np.sqrt(self.pr/self.ra)
        kappa = self.gamma*Lz**2 / np.sqrt(self.pr*self.ra)
        return nu, kappa

    def ch_params(self, T, ipath, opath, bstep, ostep, stat):
        '''Changes parameter.inp to update T, and sx '''
        with open('parameter.inp', 'r') as file:
            lines = file.readlines()

        if ostep == 0:
            ostep = int(T/self.grid.dt)

        for i, line in enumerate(lines):
            if line.startswith('idir'): #modifies input directory
                lines[i] = f'idir = "{ipath}" \n'
            if line.startswith('odir'): #modifies output directory
                lines[i] = f'odir = "{opath}" \n'
            if line.startswith('stat'): #modifies starting index
                lines[i] = f'stat = {stat}    ! last binary file if restarting an old run\n'
            if line.startswith('dt'): #modifies dt (does not change throughout algorithm)
                lines[i] = f'dt = {self.grid.dt}   ! time step\n'
            if line.startswith('step'):#modify period:
                lines[i] = f'step = {int(T/self.grid.dt)+1}      ! total number of steps\n'
            if line.startswith('cstep'): #modify cstep (bstep in current code)
                lines[i] = f'cstep = {bstep} !steps between writing global quantities\n'
            if line.startswith('tstep'): #modify tstep (ostep in current code)
                lines[i] = f'tstep = {ostep} !steps between saving fields\n'

            nu, kappa = self.get_nu_kappa()
            if line.startswith('nu'): #modifies ra (does not change throughout algorithm)
                lines[i] = f'nu = {nu}  ! kinematic viscosity\n'
            if line.startswith('kappa'): #modifies ra (does not change throughout algorithm)
                lines[i] = f'kappa = {kappa}   ! scalar difussivity\n'

        #write
        with open('parameter.inp', 'w') as file:
            file.writelines(lines)

    def inc_proj(self, fields): #TODO
        ''' Solenoidal projection of fields by using Fortran subroutines''' 
        self.write_fields(fields)

        #run specter
        subprocess.run(f'mpirun -n {self.nprocs} ./{self.solver.upper()}_PROJ', shell = True)

        #load evolved fields
        fields = self.load_fields(path='.', idx=2)
        return fields
