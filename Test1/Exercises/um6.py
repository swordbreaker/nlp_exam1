# 1 Aufgabe

from sklearn.utils.extmath import cartesian
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from verteilungen.diskrete_verteilung import *

eyes = [1,2,3,4,5,6]
omega = cartesian([eyes, eyes])

eye_sum = [sum(a) for a in omega]
eye_set = set(eye_sum)

p = np.zeros((13,))


for o in omega:
    p[sum(o)] += 1

p /= 6*6

plt.figure()
plt.bar(np.arange(13), p)
plt.show()

disk = DiskreteVerteilung(12, p[2:], list(eye_set))
disk.expected_value()
disk.variance()

# Aufgabe 3
eyes = [1,2,3,4,5,6]
omega = cartesian([eyes, eyes, eyes])

eye_sum = [sum(a) for a in omega]
eye_set = set(eye_sum)

p = np.zeros((19,))

for o in omega:
    p[sum(o)] += 1

p /= 6*6*6
# warscheinlichkeit 18 zu würfeln
p[18]
# warscheinlichkeit 17 zu würfeln
p[17]

ps = [1 - (p[18] + p[18]), p[17], p[18]]
xs = [-0.2, 4.8, 9.8]

disk = DiskreteVerteilung(18,  np.array(ps), np.array(xs))
disk.plot()
disk.expected_value()
disk.variance()

# Aufgabe 4
from verteilungen.binomial_verteilung import *

bio = BinomialVerteilung(4, 1/3)
bio.plot()

# Aufgabe 5
bio = BinomialVerteilung(8, 0.6)
#a
bio.probability(3)
#b
1 - bio.cdf(6)
#c
bio.cdf(5) - bio.cdf(2)

# Aufgabe 10
#a

p = 0.05 * 0.5 + 0.1 * 0.3 + 0.15 * 0.2
n = 6

bio = BinomialVerteilung(n, p)
print(bio.probability(1))

#b
p1 = binom_prop([1], n, 0.05)[0]
p2 = binom_prop([1], n, 0.1)[0]
p3 = binom_prop([1], n, 0.15)[0]

p = p1 * 0.5 + p2 * 0.3 + p3 * 0.2
print(p)

# Aufgabe 13
from verteilungen.poisson_verteilung import *

poisson = PoissonVerteilung(0.5)
#a
poisson.probability(3)
#b
1 - poisson.cdf(5)
#c
poisson.cdf(5) - poisson.cdf(1)

# Aufgabe 14
poisson = PoissonVerteilung(60/120)
poisson.probability(0)
poisson = PoissonVerteilung(60/30)
poisson.probability(0)

# Aufgabe 17
poisson = PoissonVerteilung(0.5)
poisson.probability(0)
1 - poisson.cdf(2)

# Aufgabe 19
from verteilungen.geometrische_verteilung import *

geom = GeometrischeVerteilung(0.5)
#a
geom.probability(3)
#b
1 - geom.cdf(5)
#c
geom.cdf(5) - geom.cdf(1)

#Aufgabe 20
geom = GeometrischeVerteilung(1/6)
geom.expected_value()

# Aufgabe 21
geom = GeometrischeVerteilung(0.3)
geom.cdf(4) - geom.cdf(1)

from verteilungen.hypergeometrische_verteilung import *

#Aufgabe 22
hyp = HypergeometrischeVerteilung(14, 5, 5)
hyp.probability(3)

#Aufgabe 24
hyp = HypergeometrischeVerteilung(100, 5, 70)
hyp.probability(3)
1 - hyp.cdf(2)