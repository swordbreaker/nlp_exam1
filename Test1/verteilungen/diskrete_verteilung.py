import numpy as np
import matplotlib.pyplot as plt

class DiskreteVerteilung(object):
    """description of class"""

    def __init__(self, n: int, ps: np.ndarray, xs: np.ndarray):
        """
            n: anzahl elemente
            p: warscheinlichkeiten
            xs: zufalsvariable (z.b. augenzahl würfel)
        """
        self.n = n
        self.ps = ps
        self.xs = xs

    def __prob(self, k):
        return self.ps[k]

    def probability(self, k):
        if(type(k) == list):
            return [self.__prob(ki) for ki in k]
        else:
            return self.__prob(k)

    def expected_value(self):
        return np.sum(self.ps * self.xs)

    def variance(self):
        e = self.expected_value()
        return np.sum(self.ps * (self.xs - e)**2)

    def plot(self):
        plt.figure()
        plt.bar(self.xs, self.ps)
        plt.show()

