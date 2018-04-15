import scipy.stats as stats
import scipy.linalg as linalg

class StandardDeviation():
    def __init__(self, avg, std):
        """ avg: loc mu, std: scale omega"""
        self.avg = avg
        self.std = std

    def cdf(self, x):
        return stats.norm.cdf(x, self.avg, self.std)

    def ppf(self, x):
        return stats.norm.ppf(x, self.avg, self.std)

    def pdf(self, x):
        return stats.norm.pdf(x, self.avg, self.std)

    def calc_z(self, x):
        return (x-self.avg)/self.std

    @staticmethod
    def calc_avg_std(p1, v1, p2, v2):
        """ 
            P(x <= v1) = p1
            P(x <= v2) = p2
            returns [avg, std]
        """
        z1 = stats.norm.ppf(p1,0,1) # (v1 - mu) / omega
        z2 = stats.norm.ppf(p2,0,1) # (v2 - mu) / omega

        print("A= ")
        print([[1, z1],[1, z2]])
        print("b = ")
        print([v1, v2])

        #  mu - z1 * omega = v1
        #  mu - z2 * omega = v2
        return linalg.solve([[1, z1],[1, z2]], [v1, v2])