import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.linalg as linalg
%matplotlib inline

#Aufgabe 1

x = np.array([3.2, 4.2, 4.1, 2.8, 3.8, 4.2, 4.5, 3.9, 2.1])
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


# Aufgabe 3

# dlaplace in scipy.stats
#a

#stats.binom.stats(n, p)

# b

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

