'''
Common nonlinearities included here
'''

import numpy as np

def cubic_stifness(kn3=1500):
    def N(_, t, y, ydot):
        out = np.zeros_like(y)
        out[0] = y[0] * kn3
        return out
    return N


