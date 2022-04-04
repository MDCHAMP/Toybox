# Toybox

![test-on-commit](https://github.com/MDCHAMP/Toybox/workflows/test-on-commit/badge.svg) [![codecov](https://codecov.io/gh/MDCHAMP/Toybox/branch/main/graph/badge.svg?token=gEJFvK2Zdw)](https://codecov.io/gh/MDCHAMP/Toybox)


Tooling to generate toy MDOF dynamics data for arbitrary physical linear and nonlinear systems in python.

Feel free to raise any issues of make suggestions.



# Quickstart guide

```python
import toybox as tb
import numpy as np

# Initialise a linear symetric sytem
system = tb.symetric(dofs=2, m=1, c=20, k=1e5)

# Define a nonlinearity
def quadratic_cubic_stiffness_2dof_single(_, t, y, ydot):
    return np.dot(y**2, np.array([5e7, 0])) + np.dot(y**3, np.array([1e9, 0]))

# Attach the nonlinearity
system.N = quadratic_cubic_stiffness_2dof_single

#Define some excitations for the system
system.excitation = [tb.forcings.white_gaussian(0, 1), None]

# Simulate
n_points = int(1e3)
fs = 1/500
normalised_data = system.simulate((n_points, fs),  normalise=True)

# Denormalise later if required
data = stem.denormalise()
```

`data` is a python `dict` with time series as follows:

Variable | Description | Dictionary key 
--- |--- | --- 
t | Time points | `'ts'` 
X<sub>d</sub>(t) | Forcing at location *d*  | `'x{d}'` 
Y<sub>d</sub>(t) | Displacement at location *d* | `'y{d}'` 
Y'<sub>d</sub>(t) | Velocity at location *d*  | `'ydot{d}'` 


# Customisation

### Arbitrary systems

`toybox.system(ndofs, M, C, K)` Allows the specification of arbitrary M, C and K matrices.

### Arbitrary forcing

Set `your_system.excitation` to a per degree-of-freedom iterable. Entries can include either:

- Premade excitations (such as `forcings.white_gaussian` or `forcings.sinusoidal`)
- Timeseries (with shape `(n_points, ndofs)`)
- `None` for unforced degrees of freedom.

