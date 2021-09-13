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

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.sim(*args, **kwargs)

    def sim(self, vector_field, w0, ts):
        raise UserWarning('No integrator selected, this is the base class')


# %% Integrators


class rk4(integrator):
    '''
    Fixed step 4th order integrator
    '''
    name = 'Runge-Kutta 4th order'
    returns = ['y', 'ydot']

    def sim(self, vector_field, w0, ts):
        self.vector_field = vector_field
        self.ts = ts
        self.dt = ts[1] - ts[0]
        self.w0 = w0
        self.dofs = len(self.w0)//2
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
            # Calculate slopes
            w = out[i]
            k1 = vf(t,  w)
            k2 = vf(t2, w + (0.5 * dt * k1))
            k3 = vf(t2, w + (0.5 * dt * k2))
            k4 = vf(t3, w + (dt * k3))
            # Update state
            out[i+1, :] = w + (k1 + 2.0*k2 + 2.0*k3 + k4) * (dt/6.0)
        # Return [y1, ydot1, y2, ydot2, ..., yd, ydotd]
        return out[:len(self.ts)].T  # Transpose for some reason


class scipy(integrator):
    '''
    Wraps the SciPy solve_IVP / odeint class for use with toybox, does not require SciPy as a dependency because the function
    is passed directly to this class constructor.
    '''

    def __init__(self, solver, args={}):
        self.solver = solver
        self.args = args

    def sim(self, vector_field, w0, ts):
        if self.solver.__name__ == 'solve_ivp':
            tspan = (min(ts), max(ts))
            self.full_res = self.solver(
                vector_field, tspan, w0, t_eval=ts, **self.args)
            return self.full_res.y
        elif self.solver.__name__ == 'odeint':
            y, self.full_res = self.solver(
                vector_field, w0, ts, tfirst=True, full_output=True, **self.args)
            return y.T

