from scipy.stats import binom

n = 4.
p = 1./3.

# Wahrscheinlichkeit:
# P(X=k) = scipy.stats.binom.pmf(k,n,p)

# kummulierte Wahrscheinlichkeit:
# P(X<=k) = scipy.stats.binom.cdf(k,n,p)

# invertierte Binomialverteilung:
# k = scipy.stats.binom.ppf(P(X<=k),n,p)

print ("* P (X = 0) = ",binom.pmf(0., n, p))
print ("* P (X = 1) = ",binom.pmf(1., n, p))
print ("* P (X = 2) = ",binom.pmf(2., n, p))
print ("* P (X = 3) = ",binom.pmf(3., n, p))
print ("* P (X = 4) = ",binom.pmf(4., n, p))

n = 8.
p = 0.6
#  P (X = 3) =?
print ("* P (X = 3) = ",binom.pmf(3., n, p))
# P (X >= 6) =?
print ("* P (X >= 6) = ",1-binom.cdf(5., n, p))
# P (2 <= X <= 5) =?
print ("* P (2 <= X <= 5) = ",binom.cdf(5., n, p)-binom.cdf(1., n, p))


from scipy.stats import poisson
#Wahrscheinlichkeit:
#P(X=k) = scipy.stats.poisson.pmf(k,lambda)

#kummulierte Wahrscheinlichkeit:
#P(X<=k) = scipy.stats.poisson.cdf(k,lambda)

#invertierte Binomialverteilung:
#k = scipy.stats.poisson.ppf(P(X<=k),lambda)

print ("Mittels Binomialverteilung (n=100, p=0.03):")
print ("P(X=0) = ",binom.pmf(0.,n,p))
print ("P(X=2) = ",binom.pmf(2.,n,p))
print ("P(X>4) = ",1-binom.cdf(4.,n,p))
print ()

lamb = n*p
print ("Mittels Poissonverteilung (lambda = n*p = 3):")
print ("P(X=0) = ",poisson.pmf(0.,lamb))
print ("P(X=2) = ",poisson.pmf(2.,lamb))
print ("P(X>4) = ",1-poisson.cdf(4.,lamb))
print ()

print ("Erwartungswerte (Anzahl Kontrollen bei 100 mal Parkieren:)")
print ("E(X) = n*p = lambda = ",lamb)
print ("Gesamtkosten mit Parkschein K_mit = 100*5 = ",n*5)
print ("Gesamtkosten ohne Parkschein K_ohne = E(X)*40 = ",lamb*40)


from scipy.stats import geom
#Wahrscheinlichkeit:
#P(X=k) = scipy.stats.geom.pmf(k,p)

#kummulierte Wahrscheinlichkeit:
#P(X<=k) = scipy.stats.geom.cdf(k,p)

#invertierte Geometrische-Verteilung:
#k = scipy.stats.geom.ppf(P(X<=k),p)

print ("Teilaufgabe a: ohne Zurücklegen!")
P_1 = 1./3.
print ("P(X=1) = ",P_1)
P_2 = 2./3.*1./2.
print ("P(X=2) = ",P_2)
P_3 = 2./3.*1./2*1
print ("P(X=3) = ",P_3)
E_a_X = 1.*P_1+2.*P_2+3.*P_3
print ("Erwartungswert E(X) = 1*P(X=1)+2*P(X=2)+3*P(X=3) = ",E_a_X)
print ()

print ("Teilaufgabe b: mit Zurücklegen!")
print ("P(X=k) = p*(1-p)^k = 1/3*(1-1/3)^(k-1)=2^(k-1)/3^k")
print ("P(X=1) = ",geom.pmf(1.,1./3.))
print ("P(X=2) = ",geom.pmf(2.,1./3.))
print ("P(X=3) = ",geom.pmf(3.,1./3.))
print ("P(X=4) = ",geom.pmf(4.,1./3.))
print ("P(X=5) = ",geom.pmf(5.,1./3.))
print ("Erwartungswert E(X) = 1/p = 1/(1/3) = 3")

from scipy.stats import hypergeom
#Wahrscheinlichkeit:
#P(X=k) = scipy.stats.hypergeom.pmf(k,N,M,n)

#kummulierte Wahrscheinlichkeit:
#P(X<=k) = scipy.stats.hypergeom.cdf(k,N,M,n)

#invertierte Hypergeometrische-Verteilung:
#k = scipy.stats.hypergeom.ppf(P(X<=k),N,M,n)

from scipy.stats import hypergeom

N = 14.
M = 5.
n = 5.
k = 3.
print ("P(X=3) = ",hypergeom.pmf(k,N,M,n))