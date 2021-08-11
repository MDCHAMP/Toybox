import pytest
import sys
import numpy as np

import toybox as tb
from toybox.premade import *
from toybox.forcings import *

systems = [symetric(2), symetric(2, m=1, c=20, k=1e3),
           underdamped_SDOF(), criticalydamped_SDOF(),
           overdamped_SDOF(), duffing_SDOF()]


def test_is_imported_correctly():
    assert 'toybox' in sys.modules

def test_requrements_satisfied():
    assert 'numpy' in sys.modules

@pytest.mark.parametrize('s', systems)
def test_square(s):
    assert s.M.shape[0] == s.M.shape[1]
    assert s.C.shape[0] == s.C.shape[1]
    assert s.K.shape[0] == s.K.shape[1]

@pytest.mark.parametrize('s', systems)
def test_simulate_simple(s):
    ins = [None for i in range(s.dofs)] 
    ins[0] = white_gaussian(0,1)
    s.excitation = ins
    s.simulate((1000, 1/200))
    for sig in s.response:
        assert sig.shape[0] == 1000
        assert not any(np.isnan(sig))

@pytest.mark.parametrize('s', systems)
def test_normalise(s):
    ins = [None for i in range(s.dofs)] 
    ins[0] = white_gaussian(0,1)
    s.excitation = ins
    s.simulate((1000, 1/100), normalise=True)
    for sig in s.response:
        assert (np.std(sig) - 1) < 10**-3
        assert (np.mean(sig) - 0) < 10**-3 
    s.denormalise()
    for sig, scale, offset in zip(s.response, s.scale, s.offset):
        assert (np.std(sig) - scale) < 10**-3
        assert (np.mean(sig) - offset) < 10**-3



def test_easy_setup():
    def quadratic_cubic_stiffness_2dof_single(_, t, y, ydot):
        return np.dot(y**2, np.array([5e7, 0])) + np.dot(y**3, np.array([1e9, 0]))

    system = tb.symetric(dofs=2, m=1, c=20, k=1e5)
    system.N = quadratic_cubic_stiffness_2dof_single
    system.excitation = [tb.forcings.white_gaussian(0, 1), None]
    system.simulate((1000, 1/500),  normalise=True)



def test_misc():
    M = np.array([[1, 0], 
                  [0, 2]])
    C = np.array([[23, -12],
                  [-8, -11]])
    K = np.array([[18, -5],
                  [-1, -23]])
    system = tb.system(M=M, C=C, K=K)
    system.N = tb.nonlinearities.cubic_stifness(2, kn3=None) 
    with pytest.raises(UserWarning):
        system._to_dict()

    with pytest.raises(AttributeError):
        system.normalise()
    
    with pytest.raises(AttributeError):
        system.denormalise()

    system.simulate((100, 1/500))
    