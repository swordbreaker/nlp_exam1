import numpy as np
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt

#würfel
dice_eye = np.array([1,2,3,4,5,6])
dices_carthesion = cartesian([dice_eye, dice_eye])

#zufallsgrösse x = Augensumme
x = {}

for d in dices_carthesion:
    sum = d[0] + d[1]
    if(sum in x):
        x[sum].append(d)
    else:
        x[sum] = [d]


# P(X = xi)
p = {}

for key, value in x.items():
    p[key] = len(value)/len(dices_carthesion)

#p = [len(value)/len(dices_carthesion) for key, value in sorted(x.items())]

# Erwartungswert = e(x) = sum( xi * f(xi))
e = {}

for xi in x.keys():
    e[xi] = xi * p[xi]

erwartungswert = np.array(list(e.values())).sum()

#varianz = (xi - mu)^^2
var = [(xi - erwartungswert)**2 for xi in x.keys()]
var2 = np.array(var) * np.array(list(p.values()))

var_sum = var2.sum()

plt.figure()
plt.bar(p.keys(), p.values())
plt.axvline(x = erwartungswert, color='r')
plt.axvline(x = erwartungswert + var_sum, color='orange')
plt.axvline(x = erwartungswert - var_sum, color='orange')
plt.show()