from scipy.special import binom
from scipy.stats import hypergeom

class HypergeometrischeVerteilung():
    """description of class"""
    def __init__(self, N, n, M):
        """ 
            N: anzahl objekte
            n: stichprobe
            M: objecte mit eigenschaft A
        """
        self.n = n
        self.N = N
        self.M = M

    def __prob(self, k):
        return (binom(self.N - self.M, self.n - k) * binom(self.M, k)) / binom(self.N, self.n)

    def probability(self, k):
        """ Wie gross ist die warscheinlichkeit das in der stichprobe k objekte mit der Eigeschaft A vorhanden sind"""
        if(type(k) == list):
            return [self.__prob(ki) for ki in k]
        else:
            return self.__prob(k)

    def expected_value(self):
        return self.n * self.M/self.N

    def variance(self):
        return self.n * self.M/self.N * (self.N - self.M)/self.N * (self.N - self.n)/(self.N - 1)

    def cdf(self, k):
        return hypergeom.cdf(k, self.N, self.M, self.n)

def binom_prop(X, n, p):
    bio = BinomialVerteilung(n, p)
    return bio.probability(X)

def example():
    N = 36 # jasskarten
    M = 4  # asse
    n = 9  # 9 karten in der hand

    hypgeo = HypergeometrischeVerteilung(N, n, M)
    print(hypgeo.probability(0)) # warscheinlichkeit 0 Asse auf der Hand
    print(hypgeo.probability([0,1,2,4]))
    print(hypgeo.expected_value())
    print(hypgeo.variance())
