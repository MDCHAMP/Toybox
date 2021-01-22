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
def test_load_and_simulate(s):
    assert s.M.shape[0] == s.M.shape[1]
    assert s.C.shape[0] == s.C.shape[1]
    assert s.K.shape[0] == s.K.shape[1]
    ts = np.linspace(0, 10, 100)
    w0 = np.zeros((2*s.dofs,))
    w0[0] = 1
    output = s.simulate(w0, ts,).T
    for sig in output:
        assert sig.shape == ts.shape
