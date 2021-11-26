import pytest
import sys
import numpy as np
from copy import deepcopy

from scipy.integrate import solve_ivp
from scipy.signal import butter, lfilter

import toybox as tb
from toybox.premade import *
from toybox.forcings import *
from toybox.nonlinearities import *
from toybox.integrators import rk4, scipy

'''
These tests are for functionality purposes only and do not check the
validity of the results
'''

# %% Define some test toys and bits and bobs

# TODO - test bad inputs to the main functions are handled graciously

npts = 1000
dt = 1/500
td = (npts, dt)

sdof = symetric(1, m=1, c=20, k=1e3)
mdof = symetric(5, m=1, c=20, k=1e3)

systems = [sdof, mdof]
nonlinearities = [None, cubic_stifness(1e3)]
forcings = [None, white_gaussian(0, 1), sinusoid(
    1, 10, np.pi), np.random.normal(0, 1, size=(npts,))]
integrators = [rk4(), scipy(solve_ivp, {'method': 'RK45'})]

M = np.array([[1, 0],
              [0, 1]])
C = np.array([[2, -1],
              [-1, 2]])
K = np.array([[20, -10],
              [-10, 20]])

# Helper functions


def _no_nans(data):
    res = True
    for key, array in data.items():
        res = res and not any(np.isnan(array))
    return res

# %% some very basic tests now


def test_is_imported_correctly():
    assert 'toybox' in sys.modules


def test_requrements_satisfied():
    assert 'numpy' in sys.modules


# %% Some interface tests now

def test_readme_recipe():
    # Initialise a linear symmetric system
    system = tb.symetric(dofs=2, m=1, c=20, k=1e5)
    # Define a nonlinearity

    def quadratic_cubic_stiffness_2dof_single(_, t, y, ydot):
        return np.dot(y**2, np.array([5e7, 0])) + np.dot(y**3, np.array([1e9, 0]))
    # Attach the nonlinearity
    system.N = quadratic_cubic_stiffness_2dof_single
    # Define some excitations for the system
    system.excitation = [tb.forcings.white_gaussian(0, 1), None]
    # Simulate
    n_points = 1000
    fs = 1/500
    normalised_data = system.simulate((n_points, fs),  normalise=True)
    # de-normalise later if required
    data = system.denormalise()
    _no_nans(data)


def test_ts_vector_input():
    system = tb.symetric(dofs=2, m=1, c=20, k=1e5)
    system.excitation = [tb.forcings.white_gaussian(0, 1), None]
    data = system.simulate(np.linspace(0, npts*dt, npts),  normalise=True)
    assert _no_nans(data)    

@pytest.mark.parametrize('I', integrators)
def test_zero_solution(I):
    system = tb.symetric(dofs=2, m=1, c=20, k=1e5)
    data = system.simulate(td,  integrator=I, normalise=True)
    assert _no_nans(data)


def test_linear_modes():
    system = tb.symetric(dofs=2, m=1, c=20, k=1e5)
    es, evs = system.linear_modes()
    assert len(es) == system.dofs
    assert evs.shape == system.M.shape

# %% Test that bad inputs / code are caught


def test_errors_before_simulation():
    system = tb.system(M=M, C=C, K=K)
    system.N = cubic_stifness()
    with pytest.raises(UserWarning):
        system._to_dict()
    with pytest.raises(AttributeError):
        system.normalise()
    with pytest.raises(AttributeError):
        system.denormalise()


def test_no_integrator():
    with pytest.raises(UserWarning):
        tb.integrators.integrator()(None, None, None)


# %% Test various configurations of system / nonlinearity / forcing

@pytest.mark.parametrize('S', systems)
@pytest.mark.parametrize('N', nonlinearities)
@pytest.mark.parametrize('F', forcings)
@pytest.mark.parametrize('I', integrators)
def test_configurations(S, N, F, I):
    S = deepcopy(S)
    S.N = N
    S.excitation = [None]*S.dofs
    S.excitation[0] = F
    data = S.simulate(td, integrator=I)
    _no_nans(data)


# %% Test normalisation / de-normalisation works correctly

@pytest.mark.parametrize('S', systems)
def test_normalisation(S):
    S = deepcopy(S)
    S.excitation = [None]*S.dofs
    S.excitation[0] = white_gaussian(0, 1)
    data = S.simulate(td, normalise=True)
    assert _no_nans(data)
    for sig in S.response:
        assert (np.std(sig) - 1) < 1e-3
        assert (np.mean(sig) - 0) < 1e-3
    S.denormalise()
    for sig, scale, offset in zip(S.response, S.scale, S.offset):
        assert (np.std(sig) - scale) < 1e-3
        assert (np.mean(sig) - offset) < 1e-3


@pytest.mark.parametrize('S', systems)
@pytest.mark.parametrize('w', [10,20,50,80])
def test_scipy_filter(S, w):
    fs = 500
    normal_cutoff = w / (0.5 * fs)
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    
    S = deepcopy(S)
    S.excitation = [None]*S.dofs

    x = white_gaussian(0, 1).scipy_filter(a, b, lfilter)
    assert hasattr(x, '_filt')
    S.excitation[0] = x

    


