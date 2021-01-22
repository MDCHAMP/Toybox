import pytest
import sys
import numpy as np

from toybox.premade import *

systems = [symetric(2), symetric(1, 1, 1),
           underdamped_SDOF(), criticalydamped_SDOF(),
           overdamped_SDOF(), duffing_SDOF()]


def test_is_imported_correctly():
    assert 'toybox' in sys.modules


def test_requrements_satisfied():
    assert 'numpy' in sys.modules



@pytest.mark.parametrize('s', systems)
def test_load(s):
    assert s.M.shape[0] == s.M.shape[1]
    assert s.C.shape[0] == s.C.shape[1]
    assert s.K.shape[0] == s.K.shape[1]

@pytest.mark.parametrize('s', systems)
def test_simulate(s):
    ts = np.linspace(0, 10, 1000)
    w0 = np.zeros((2*s.dofs,))
    w0[0] = 1
    s.simulate(w0, ts)
    for sig in s.response:
        assert sig.shape == ts.shape
        assert not any(np.isnan(sig))

@pytest.mark.parametrize('s', systems)
def test_normalise(s):
    ts = np.linspace(0, 10, 1000)
    w0 = np.zeros((2*s.dofs,))
    w0[0] = 1
    s.simulate(w0, ts)
    s.normalise()
    for sig in s.response:
        assert (np.std(sig) - 1) < 10**-3
        assert (np.mean(sig) - 0) < 10**-3 
    s.denormalise()
    for sig, scale, offset in zip(s.response, s.scale, s.offset):
        assert (np.std(sig) - scale) < 10**-3
        assert (np.mean(sig) - offset) < 10**-3