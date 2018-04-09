import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class StochStats(object):
    """description of class"""

    def __init__(self, l:[float]):
        self.l = np.array(l)
        self.sum = np.sum(l)
        self.rel_l = l/self.sum

    def mean(self, rel=False):
        if rel:
            return self.rel_l.mean()
        else:
            return self.l.mean()

    def median(self, rel=False):
        if rel:
            return np.median(self.rel_l)
        else:
            return np.median(self.l)

    def std_emp(self, rel=False):
        """empirische standartabweichung"""
        if rel:
            return  self.rel_l.std(ddof=0)
        else:
            return  self.l.std(ddof=0)

    def std_theo(self, rel=False):
        """theoretische standartabweichung"""
        if rel:
            return self.rel_l.std(ddof=1)
        else:
            return self.l.std(ddof=1)

    def quartil(self, p:[float] = [25, 75], rel=False):
        """ p: a list of precantages eg [25, 75] """
        if rel:
            return np.percentile(self.rel_l, p)
        else:
            return np.percentile(self.l, p)

    def box_plot(self, rel=False):
        if rel:
            plt.figure()
            plt.boxplot(self.rel_l, vert=False)
            plt.show()
        else:
            plt.figure()
            plt.boxplot(self.l, vert=False)
            plt.show()