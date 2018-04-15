import  math
from scipy.stats import poisson

class PoissonVerteilung():
    """description of class"""

    @staticmethod
    def from_n_and_p(n, p):
        return PoissonVerteilung(n * p)

    def __init__(self, lambda_):
        """ lambda_ : mittlerer interval """
        self.lambda_ = lambda_

    def __prob(self, k):
        return (self.lambda_ ** k) / math.factorial(k) * math.e ** -self.lambda_

    def probability(self, k):
        if(type(k) == list):
            return [self.__prob(ki) for ki in k]
        else:
            return self.__prob(k)
        
    def expected_value(self):
        return self.lambda_

    def variance(self):
        return self.lambda_

    def cdf(self, k):
        return poisson.cdf(k, self.lambda_)

def poisson_prop(X, _lambda):
    poisson = PoissonVerteilung(_lambda)
    return poisson.probability(X)

def example():
    p = 0.03 # 3% chance auf kontrolle
    n = 100  # auf 100 mal parkieren

    poisson = PoissonVerteilung.from_n_and_p(n, p)
    print(poisson.probability(0))  # warscheinlichkeit kein mal kontrolliert zu werden
    print(poisson.probability([0,1,2,3,4,5]))

    print(poisson.expected_value())    
    print(poisson.variance())