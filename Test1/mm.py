import numpy as np
from scipy import linalg
from graphviz import Digraph

class MM():
    def __init__(self, start: np.ndarray, u: np.ndarray):
        self.start = start
        self.u = u
        
    def sample(self, n):
        pi = []
        pi.append(self.start)

        for i in range(n):
            pi.append(np.dot(pi[-1], self.u))

        return pi

    def grenzverteilung(self):
        r, c = self.u.shape
        A = np.transpose(self.u) - np.eye(r)

        for k in range(c):
            A[r-1, k] = 1

        b = np.zeros(shape=(r, 1))
        b[r-1, 0] = 1
        print('Gleichungssystem:')
        print('A:')
        print(A)
        print('b:')
        print(b)
        print()
        return linalg.solve(A,b)

    def draw_graph(self):
        n = len(self.start)
        dot = Digraph()
        for i in range(n):
            dot.node(str(i), str(i))
            for k in range(n):
                if(self.u[i,k] != 0):
                    dot.edge(str(i), str(k), label=f"{self.u[i,k]:2.2f}")
        dot.render('graphviz/hmm.gv', view=True)

    def absorb_values(self, absorb_index):
        """ absorb_index index of the absorb node """

        U = self.u
        r, c = self.u.shape
        A = np.eye(r-1)
        b = np.zeros(shape=(r-1, 1))

        k = 0
        for i in range(r):
            if i == absorb_index: 
                k = 1
                continue
            s = f"p{i} = "
            l = 0
            for j in range(c):
                if i == j: 
                    if(U[i,j] == 1): continue
                    if(U[i,j] != 0):
                        A[i-k,j-l] = 1 - U[i,j]
                        s += f"{1 - U[i,j]} * p{j} + "
                    continue
                if j == absorb_index:
                    b[i-k] = U[i,j]
                    l = 1
                    continue
                if(U[i,j] != 0):
                        A[i-k,j-l] = -U[i,j]
                        s += f"{U[i,j]} * p{j} + "
            
            print(s)

        print("A = ")
        print(A)
        print("b = ")
        print(b)
        return linalg.solve(A,b)

    def absorb_lenght(self, absorb_index):
        """ absorb_index index of the absorb node """

        U = self.u
        r, c = self.u.shape
        A = np.eye(r-1)
        b = np.ones(shape=(r-1, 1))

        k = 0
        for i in range(r):
            if i == absorb_index: 
                k = 1
                continue
            s = f"m{i} = 1 + "
            l = 0
            for j in range(c):
                if i == j: 
                    if(U[i,j] == 1):
                        b[i-k] = 0
                    elif(U[i,j] != 0):
                        A[i-k,j-l] = 1 - U[i,j]
                        s += f"{1 - U[i,j]} * m{j} + "
                    continue
                if j == absorb_index:
                    l = 1
                    continue
                if(U[i,j] != 0):
                        A[i-k,j-l] = -U[i,j]
                        s += f"{U[i,j]} * m{j} + "
            
            print(s)

        print("A = ")
        print(A)
        print("b = ")
        print(b)
        return linalg.solve(A,b)

def example():
    u = [[0  , 1/3, 1/3, 1/3],
         [1/3,   0, 1/3, 1/3],
         [1/3, 1/3,   0, 1/3],
         [1/3, 1/3, 1/3, 0  ],
         ]
    u = np.array(u)

    start = np.array([0, 0, 0, 1])
    
    mm = MM(start, u)
    print(mm.sample(10))

    print(mm.grenzverteilung())


    u = [
        [1  ,   0,  0,   0,   0,   0],
        [0.4,   0,0.6,   0,   0,   0],
        [0.4,   0,  0,   0, 0.6,   0],
        [0  , 0.4,  0,   0,   0, 0.6],
        [0  ,   0,  0, 0.4,   0, 0.6],
        [0  ,   0,  0,   0,   0,   1],
        ]
    u = np.array(u)
    start = np.array([0,1,0,0,0,0])

    mm = MM(start, u)
    print(mm.absorb_values(5))

    print(mm.absorb_lenght(5))

    mm.draw_graph()