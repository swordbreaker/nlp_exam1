import numpy as np
from sklearn.utils.extmath import cartesian


sums = [sum(a) for a in cartesian([np.arange(1,10), np.arange(1,10)])]
omega_b = set(sums)

prods = [a[0] * a[1] for a in cartesian([np.arange(1,10), np.arange(1,10)])]
omega_c = set(prods)