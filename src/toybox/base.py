'''
Base classes for toybox
'''

import warnings
import numpy as np

from toybox.integrators import integrator, rk4
from toybox.forcings import shaker

# %% Notation reference


_Notation = {
    # Misc.
    'n': 'number of points',
    'd': 'number of DOFS',
    't': 'time (float)',
    'ts': 'time vector (vector)',
    'w': 'system state (vector 2dx1)',
    'ws': 'system state (vector 2dxn)',
    'w0': 'initial condition (vector 2dx1)',
    # Variables.
    'x': 'system input (vector dx1)',
    'y': 'displacement (vector dx1)',
    'ydot': 'velocity (vector dx1)',
    'yddot': 'acceleration (vector dx1)',
    # Time signals.
    'xs': 'system input (vector dxn)',
    'ys': 'displacement (vector dxn)',
    'ydots': 'velocity (vector dxn',
    'yddots': 'acceleration (vector dxn)',
    # System parameters.
    'M': 'mass matrix (square dxd)',
    'C': 'damping matrix (square dxd)',
    'K': 'stiffness matrix (square dxd)',
    # Nonlinearity.
    'N': 'nonlinearity (callable N(t, y, ydot) -> (vector dx1))',
}


# %% Base class


class system():
    '''
    MDOF system
    '''
    name = 'generic system'

    def __init__(self, M=None, C=None, K=None, N=None):
        self.M = M
        self.C = C
        self.K = K
        self.N = N
        self.dofs = M.shape[0]
        self.slopes = None
        self.excitation = [None] * self.dofs

    def get_slopes(self, x_t=None):
        if x_t is None: x_t = lambda t: np.zeros((self.dofs,)) #Handle the no forcing case
        if self.N is None: self.N =lambda _, t, _y, _ydot: np.zeros((self.dofs,)) # Handle the non nonlinearity case
        
        def vector_field(t, w):
            x = x_t(t) # Get x from x_t
            # Recover displacements and velocities from current timestep
            _y = w[0::2]
            _ydot = w[1::2]
            # Calculate the derivatives of _y and _ydot and interleave vectors to form d w / dt
            dw = np.zeros_like(w) # pre-allocate 
            dw[0::2] = _ydot
            dw[1::2] = np.linalg.inv(self.M) @ (x - (self.C@_ydot) - self.K@_y - self.N(self, t, _y, _ydot))
            return dw
        
        return vector_field

    def simulate(self, ts, x_t=None,  w0=None, integrator=None, normalise=False):
        #  Time vector: self.ts ndarray (n, 1)
        if type(ts) is tuple:
            npts, dt = ts
            self.ts = np.linspace(0, npts*dt, npts)
        else:
            npts = len(ts)
            fs   = 1 / (ts[1] - ts[0])   
        # Excitations: self.x_t callable x_t(t) -> x
        if x_t is None:
            try:
                self.shaker = shaker(self.excitation)
            except AttributeError:
                warnings.warn(
                    'No Excitations provided proceeding without external forcing', UserWarning)
                self.shaker = shaker([None]*self.dofs)
            self.xs = self.shaker.generate(self.ts) # Save the forcing history
            x_t = self.shaker.get_x_t(self.ts, self.xs)
        else:
            self.xs = np.array([x_t(t) for t in self.ts])[:, None] # Add a dummy index to make the integrators happy later
        # ICs
        if w0 is None:
            warnings.warn(
                'No initial conditions provided proceeding with zero initial condition', UserWarning)
            w0 = np.zeros((self.dofs*2,))
        # d w / dt
        if self.slopes is None:
            self.slopes = self.get_slopes(x_t)
        # Integrator
        if integrator is None:
            integrator = rk4()
        # Simulate
        self.response = integrator.sim(self.slopes, w0, self.ts)
        if normalise:
            self.normalise()
        return self._to_dict()

    def normalise(self):
        # Recovering the offset and scale params so we can recover original after analysis
        try:
            self.offset = np.mean(self.response, axis=1)
            self.scale = np.std(self.response, axis=1)
            self.response = ((self.response.T - self.offset) / self.scale).T
        except AttributeError:
            raise AttributeError('Not simulated yet')
        return self._to_dict()

    def denormalise(self):
        # Recovering the original signal
        try:
            self.response = ((self.response.T * self.scale) + self.offset).T
        except AttributeError:
            raise AttributeError('Not normalised yet')

        return self._to_dict()

    def linear_modes(self):
        # Return (eigenvectors, eigenvalues) of the underlying linear system (No damping)
        return np.linalg.eig(np.dot(np.linalg.inv(self.M), self.K))

    def linear_exact(self, ts):
        # Return exact (within numerical tolerance) solution of underlying linear system
        raise NotImplementedError

    def _to_dict(self):
        if not hasattr(self, 'response'):
            raise UserWarning('System not simulated cannot produce results. Call System.simulate()')
        ys = self.response[0::2]
        ydots = self.response[1::2]
        out = {}
        out['t'] = self.ts
        for doff, x, y, ydot in zip(range(1, self.dofs+1), self.xs.T, ys, ydots):
            out[f'x{doff}'] = x
            out[f'y{doff}'] = y
            out[f'ydot{doff}'] = ydot
        return out

