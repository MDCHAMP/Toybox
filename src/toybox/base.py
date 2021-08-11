'''
Base classes for toybox
'''

import warnings
import numpy as np

from toybox.integrators import rk4
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

    def get_slopes(self):
        def vector_field(w, t, x):
            # Recover displacements and velocities from current timestep
            _y = w[0::2]
            _ydot = w[1::2]
            # Calculate field = [ydot1, yddot1, ydot2, yddot2, ..., ydotd, yddotd] etc...
            m_inv = np.linalg.inv(self.M),
            damping = np.dot(self.C, _ydot)
            stiffness = np.dot(self.K, _y)
            if self.N is None:
                nonlinearity = np.zeros_like(x)
            else:
                nonlinearity = self.N(self, t, _y, _ydot)
            lhs = (x - damping - stiffness - nonlinearity)
            lhs = np.dot(m_inv, lhs)
            ydot = _ydot
            yddot = lhs
            # Interleave vectors to reform state vector
            field = np.empty((ydot.size + ydot.size,), dtype=ydot.dtype)
            field[0::2] = ydot
            field[1::2] = yddot
            return field
        return vector_field

    def simulate(self, ts, w0=None, xs=None, integrator=None, normalise=False):
        if type(ts) is tuple:
            npts, fs = ts
            self.ts = np.linspace(0, npts*fs, npts)
        if xs is None:
            try:
                self.xs = shaker(self.excitation).generate(npts)
            except AttributeError:
                warnings.warn(
                    'No Excitations provided proceeding without forcing', UserWarning)
                self.xs = shaker([None]*self.dofs).generate(npts)
        if w0 is None:
            warnings.warn(
                'No initial conditions provided proceeding without', UserWarning)
            w0 = np.zeros((self.dofs*2,))
        if self.slopes is None:
            self.slopes = self.get_slopes()
        if integrator is None:
            integrator = rk4
        self.response = integrator(self.slopes, w0, self.ts, self.xs)()
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
            raise UserWarning('System not simulated cannot produce dict')
        ys = self.response[0::2]
        ydots = self.response[1::2]
        out = {}
        out['t'] = self.ts
        for doff, x, y, ydot in zip(range(1, self.dofs+1), self.xs.T, ys, ydots):
            out[f'x{doff}'] = x
            out[f'y{doff}'] = y
            out[f'ydot{doff}'] = ydot
        return out


# %%
