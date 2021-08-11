'''
submodule for integrators that generate time series

API TBC

Only IC problems considered for now
'''


import numpy as np


# %% Base class


class integrator():
    '''
    Base class for integrators
    '''
    name = 'integrator'
    returns = ['y', 'ydot', 'yddot']

    def __init__(self, vector_field, w0, ts, xs=None):
        self.vector_field = vector_field
        self.ts = ts
        self.dt = ts[1] - ts[0]
        self.w0 = w0
        self.dofs = len(self.w0)//2
        if xs is None: self.xs = np.zeros((ts.shape[0]+10, self.dofs))
        else: self.xs = xs

    def get_x(self, t):
        '''
        Interpolate for the forcing between timesteps
        '''
        x = np.zeros((self.dofs, ))
        for dof in range(self.dofs):
            x[dof] = np.interp(t, self.ts, self.xs[:,dof])

        return x

    def __call__(self):
        return self.sim()

    def sim(self):
        raise NotImplementedError('No integrator selected')


# %% Integrators


class rk4(integrator):
    '''
    Fixed step 4th order integrator
    '''
    name = 'Runge-Kutta 4th order'
    returns = ['y', 'ydot']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sim(self):
        '''
        Perform simulation
        '''
        out = np.zeros((len(self.ts)+1, len(self.w0)))
        out[0, :] = np.array(self.w0)
        dt = self.dt
        vf = self.vector_field
        # Main loop
        for i, t in enumerate(self.ts[:]):
            # Get time steps
            t1 = t
            t2 = t + 0.5*dt
            t3 = t + dt
            # Get forces
            x1 = self.get_x(t1)
            x2 = self.get_x(t2)
            x3 = self.get_x(t3)
            # Calculate slopes
            w = out[i]
            k1 = vf(w, t, x1)
            k2 = vf(w + (0.5 * dt * k1), t2, x2)
            k3 = vf(w + (0.5 * dt * k2), t2, x2)
            k4 = vf(w + (dt * k3),       t3, x3)
            # Update state
            out[i+1, :] = w + (k1 + 2.0*k2 + 2.0*k3 + k4) * (dt/6.0)
        # Return [y1, ydot1, y2, ydot2, ..., yd, ydotd]
        return out[:len(self.ts)].T # Transpose for some reason
