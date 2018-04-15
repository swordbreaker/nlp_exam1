from scipy.stats import geom

class GeometrischeVerteilung():
    """description of class"""

    def __init__(self, p):
        """ X: list of numbers """
        self.p = p

    def __prob(self, k):
        return self.p * (1 - self.p)**(k - 1)

    def probability(self, k):
        if(type(k) == list):
            return [self.__prob(ki) for ki in k]
        else:
            return self.__prob(k)

    def expected_value(self):
        return 1 / self.p

    def variance(self):
        return (1 - self.p)/(self.p ** 2)

    def cdf(self, k):
        return geom.cdf(k, self.p)

def geo_prop(X, p):
    geo = GeometrischeVerteilung(p)
    return geo.probability(X)

def example():
    # 3 schlüssel am schlüsselbund
    # versucht 1 schlüssel um die tür zu öffnen
    # annahme merks sich nicht welcher schlüssel er schon getestet hat

    p = 1/3
    geo = GeometrischeVerteilung(p)
    print(geo.probability(1)) #Warscheinlichkeit tür nach einem versuch offen
    print(geo.probability([1, 2, 3, 4, 5]))
    print(geo.expected_value())
    print(geo.variance())