import numpy as np
from mm import *

# Aufgabe 1

#a
p = 0.8 * 0.6 + 0.4 * 0.1
print(p)

#b


# Aufgabe 6
label = ['R', 'P', 'W1']

U = np.array([
        [.5, .5, 0],
        [.25, .5, .25],
        [0, .5, .5]
    ])

s = np.array([4000, 4000, 4000])

mm = MM(s, U)
mm.sample(1)

mm.sample(2)

s = np.array([0, 6000, 6000])
mm = MM(s, U)
mm.sample(3)

# Aufgabe 10
U = np.array([
        [1, 0, 0],
        [.1, .2, .7],
        [.2, .3, .5]
    ])

mm = MM([1/3, 1/3, 1/3], U)
mm.draw_graph()
mm.absorb_values(0)