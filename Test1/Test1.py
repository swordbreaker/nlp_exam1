import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.linalg as linalg
import stoch_stats
import standard_deviation
%matplotlib inline

#Aufgabe 1

x = np.array([3.2, 4.2, 4.1, 2.8, 3.8, 4.2, 4.5, 3.9, 2.1])

s = stoch_stats.StochStats(x)
print(s.mean())
print(s.median())
print(s.std_emp())
print(s.std_theo())
print(s.quartil())
s.box_plot()


mean = x.mean()
median = np.median(x)
std_emp = x.std(ddof=0) #empirische
std_theo = x.std(ddof=1) #theoretische

print(f"mean:\t {mean}")
print(f"median: {median}")
print(f"std empirische: {std_emp}")
print(f"std theoretische: {std_theo}")

#quartiele
print(np.percentile(x, [25, 75]))

plt.figure()
plt.boxplot(x)
plt.show()

# p * mean(x) + x_10 >= 10 * 3.75
# == 4.7


#Aufgabe 2

#a
mu = 2
omega = 2

print(stats.norm.cdf(1, mu, omega))
print(stats.norm.cdf(4, mu, omega) - stats.norm.cdf(2, mu, omega))

#b
mu = 25
omega = 5
stats.norm.ppf(0.9, mu, omega)

#c
# P(x <= 55kg) = 0.25
# P(x >= 85kg) = 0.1

# Z = (x - mu) / omega

z1 = stats.norm.ppf(0.25,0,1) # (55 - mu) / omega
z2 = stats.norm.ppf(0.9,0,1)  # (85 - mu) / omega

#  mu - z1 * omega = 55
#  mu - z2 * omega = 85


linalg.solve([[1, z1],[1, z2]], [55, 85])

standard_deviation.StandardDeviation.calc_avg_std(0.25, 55, 0.9, 85)

from proba_tree import *
from verteilungen.diskrete_verteilung import *
# Aufgabe 3

#a
n1 = Node(1)
n2 = Node(0)
n3 = Node(1/2, childs=[n1, n2])
n4 = Node(1/2)
n5 = Node(2/3, childs=[n3, n4])
n6 = Node(1/3)
n7 = Node(3/4, childs=[n6,n5])
n8 = Node(1/4)
n9 = Node(4/5, childs=[n8,n7])
n10 = Node(1/5)
n11 = Node(5/6, childs=[n10,n9])
n12 = Node(1/6)
n13 = Node(0, childs=[n12, n11], is_root=True)

n13.show()

ps = [
    n12.prop_up(),
    n10.prop_up(),
    n8.prop_up(),
    n6.prop_up(),
    n4.prop_up(),
    n1.prop_up(),
    ]

xs = [1,2,3,4,5,6]

disk = DiskreteVerteilung(6, np.array(ps), np.array(xs))
print(disk.expected_value())
print(disk.variance())

#b
n1 = Node(0.03, label="_OK")
n3 = Node(0.04, label="_OK")
n5 = Node(0.02, label="_OK")
n2 = Node(0.97, label="OK")
n4 = Node(0.96, label="OK")
n6 = Node(0.98, label="OK")

m1 = Node(0.5, childs=[n1, n2], label="M1")
m2 = Node(0.25, childs=[n3, n4], label="M2")
m3 = Node(0.25, childs=[n5, n6], label="M3")

root = Node(0, childs=[m1, m2, m3], is_root=True)

b1 = root.prop_label("_OK")
print(b1)
root.show()

b2 = (n1.weight * m1.weight) / root.prop_label("_OK")
print(b2)

#Aufgabe 5

p = [[0  , 1/3, 1/3, 1/3],
     [1/3,   0, 1/3, 1/3],
     [1/3, 1/3,   0, 1/3],
     [1/3, 1/3, 1/3, 0  ],
     ]
p = np.array(p)
pi0 = np.array([0, 0, 0, 1])
print(np.dot(pi0, p))

#Grenzverteilung




#Aufgabe 6


#a
from mm import *
u = [
    [0, 0.01, 0.74, 0.21, 0.04, 0.00],
    [0, 0.18, 0.32, 0.20, 0.12, 0.18],
    [0, 0.53, 0.01, 0.42, 0.02, 0.02],
    [0, 0.01, 0.11, 0.05, 0.61, 0.22],
    [0, 0.05, 0.33, 0.42, 0.02, 0.18],
    [0, 0.00, 0.00 ,0.00, 0.00, 1.00],
    ]
u = np.array(u)

start = [1, 0, 0, 0, 0, 0]
start = np.array(start)

mm = MM(start, u)

mm.draw_graph()
mm.absorb_lenght(5)

#b

from hmm import *
from sklearn.utils.extmath import cartesian

U = [
    [0.4, 0.6],
    [0.8, 0.2]
    ]
U = np.array(U)

E = [
    [0.5, 0.4, 0.1],
    [0.2, 0.2, 0.6]
    ]
E = np.array(E)
p0 = np.array([[0.2, 0.8]])
o = np.array([[0,1,2]])
hmm = Hmm(U, E, p0,o)

max = 0
comb = None
labels = ['a', 'b', 'c']
for a in cartesian([[0,1,2],[0,1,2]]):
    p, _ = hmm.forward(np.array([a]))
    if(p > max):
        max = p
        comb = a

print(max)
print(f"{labels[comb[0]]} {labels[comb[1]]}")

p, a = hmm.viterbi(np.array([[0,1,0]]))
print(p)
print(f"{labels[int(a[0])]} {labels[int(a[1])]} {labels[int(a[2])]}")