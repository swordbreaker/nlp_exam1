import numpy as np
from hmmlearn import hmm
from graphviz import Digraph

class Hmm(object):
    """description of class"""

    def __init__(self, start, p):
        n = len(start)
        self.model = hmm.GaussianHMM(n_components=n)
        self.model.startprob_ = start
        self.model.transmat_ = p
        self.model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
        self.model.covars_ = np.tile(np.identity(2), (3, 1, 1))

        self.dot = Digraph()
        
        for i in range(n):
            self.dot.node(str(i), str(i))
            for k in range(n):
                if(p[i,k] != 0):
                    self.dot.edge(str(i), str(k), label=f"{p[i,k]:2.2f}")

    def show(self):
        self.dot.render('graphviz/hmm.gv', view=True)

    def sample(self, n):
        X, Z = self.model.sample(100)
        return (X, Z)



p = [[0  , 1/3, 1/3, 1/3],
    [1/3,   0, 1/3, 1/3],
    [1/3, 1/3,   0, 1/3],
    [1/3, 1/3, 1/3, 0  ],
    ]
p = np.array(p)
pi = np.array([0, 0, 0, 1])

hmm = Hmm(pi, p)
hmm.show()


