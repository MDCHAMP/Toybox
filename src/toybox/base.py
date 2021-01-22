'''
Base classes for toybox
'''

import numpy as np

from toybox.integrators import rk4

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
            damping = np.dot(self.C,_ydot)
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

    def simulate(self, w0, ts, xs=None, integrator=None):
        if self.slopes is None:
            self.slopes = self.get_slopes()
        if integrator is None:
            integrator = rk4
        self.response = integrator(self.slopes, w0, ts, xs).sim()
        return self.response

    def normalise(self):
        # Recovering the offset and scale params so we can recover original after analysis
        self.offset = np.mean(self.response, axis=0)
        self.scale = np.std(self.response, axis=0)
        self.response = (self.response - self.offset) / self.scale
        return self.response

    def denormalise(self):
        # Recovering the original signal
        try:
            self.response = (self.response * self.scale) + self.offset
        except AttributeError:
            # Not normalised yet
            pass
        return self.response

    def linear_modes(self):
        # Return (eigenvectors, eigenvalues) of the underlying linear system (No damping)
        return np.linalg.eig(np.dot(np.linalg.inv(self.M), self.K))

    def linear_exact(self, ts):
        # Return exact (within numerical tolerance) solution of underlying linear system
        raise NotImplementedError
