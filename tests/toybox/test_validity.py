import pytest
import sys
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.integrate import solve_ivp, odeint


from toybox.premade import *
from toybox.forcings import *
from toybox.nonlinearities import *
from toybox.integrators import rk4, scipy

'''
Some tests that make sure that the simulations produce valid results
'''

PLOT = False
TOL = 1e-6

# %% Define some test toys and bits and bobs

# TODO - test problems with different sampling frequencies and integration lengths based on numerical stiffness

npts = 1000
dt = 1/500
td = (npts, dt)


def _no_nans(data):
    res = True
    for key, array in data.items():
        res = res and not any(np.isnan(array))
    return res


def _nmse(y, y_hat):
    return 100 * (1/(len(y) * np.std(y))) * sum((y - y_hat)**2)


t1 = {
    'system': symetric(1, m=1, c=-6, k=9, N=None),
    'w0': np.array([0, 2]),
    'x_t': None,
    'y_t': lambda t: 2*t*np.exp(3*t)
}

t2 = {
    'system': symetric(1, m=1, c=0, k=100, N=None),
    'w0': np.array([0, 10]),
    'x_t': None,
    'y_t': lambda t: np.sin(10*t)
}

t3 = {
    'system': symetric(1, m=1, c=0.2, k=1, N=None),
    'w0': np.array([0, 1]),
    'x_t': lambda t: -0.2*np.exp(-t/5)*np.cos(t),
    'y_t': lambda t: np.exp(-t/5)*np.sin(t)
}

ODEs = [t1, t2, t3]
integrators = [rk4(), scipy(
    solve_ivp, {'method': 'RK45', 'rtol': TOL}), scipy(odeint)]

# %% Test that the sims are valid to tolerance


@pytest.mark.parametrize('I', integrators)
@pytest.mark.parametrize('T', ODEs)
def test_valid_results(I, T):
    I = deepcopy(I)
    system = deepcopy(T['system'])
    data = system.simulate(td, w0=T['w0'], x_t=T['x_t'], integrator=I)
    y_hat = data['y1']
    y = T['y_t'](np.linspace(0, npts*dt, num=npts))
    t = data['t']

    if PLOT:
        plt.figure()
        plt.plot(t, y)
        plt.plot(t, y_hat)
        plt.legend(['True', 'Simulated'])
        plt.show()
    assert _no_nans(data)
    assert _nmse(y, y_hat) < TOL
