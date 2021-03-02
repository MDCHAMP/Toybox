'''
Common nonlinearities included here
'''

import numpy as np

def cubic_stifness(dofs, kn3=None):
    if kn3 is None:
        kn3 = np.zeros((dofs, dofs))
        kn3[0,0] = 1500    
    def N(self, t, y, ydot):
        return np.dot(kn3, y**3)
    return N

