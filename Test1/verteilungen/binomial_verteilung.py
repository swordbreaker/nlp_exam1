from scipy.special import binom
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class BinomialVerteilung():
    """description of class"""
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def __prob(self, k):
        return binom(self.n, k) * self.p ** k * (1 - self.p)**(self.n-k)

    def probability(self, k):
        if(type(k) == list):
            return [self.__prob(ki) for ki in k]
        else:
            return self.__prob(k)

    def plot(self):
        ns = np.arange(self.n+1)
        ps = [self.__prob(n) for n in ns]
        plt.figure()
        plt.bar(ns, ps)
        plt.show()

    def expected_value(self):
        return self.n * self.p

    def variance(self):
        return self.n * self.p * (1 - self.p)

    def cdf(self, k):
        return stats.binom.cdf(k, self.n, self.p)

def binom_prop(X, n, p):
    bio = BinomialVerteilung(n, p)
    return bio.probability(X)

def example():
    # n = 3   (Fragen)
    # p = 1/3 (Trefferwarscheinlichkeit)
    # X anzahl korrekt gelöster Fragen
    b = BinomialVerteilung(3, 1/3)
    print(b.probability(0))
    print(b.probability([0,1,2,3]))

    #or
    print(binom_prop([0,1,2,3], 3, 1/3))