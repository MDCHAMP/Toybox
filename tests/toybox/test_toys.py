import pytest
import numpy as np

import toybox

def test_is_imported_correctly():
    assert type(toybox.__version__) is str


    