from hmm import *

U = np.array([
        [.9, .1],
        [.7, .3]
    ])

E = np.array([
        [.5, .5],
        [.75, .25]
    ])

s = np.array([[.6, .4]])

hmm = Hmm(U, E, s, np.array([[0,0,0]]), ['F', 'U'], ['K', 'Z'])

for i in range(2, 11):
    print(hmm.best_o(i))

hmm.viterbi(np.array([[1,1,1]]))
hmm.viterbi(np.array([[0,0,0]]))
hmm.viterbi(np.array([[1,0,1,0,1,0]]))

#Aufgabe 4

hidden_labels = ['X1', 'X2', 'X3']
obseve_labels = ['Y1', 'Y2']

U = np.array([
        [.1, .2, .7],
        [.2, .7, .1],
        [.7, .1, .2],
    ])

E = np.array([
    [.9, .1],
    [.1, .9],
    [0 ,  1],
    ])

start = np.array([[1/3,1/3,1/3]])

hmm = Hmm(U, E, start, np.array([[0,0,0]]), hidden_labels, obseve_labels)

for i in range(2, 11):
    print(hmm.best_o(i))

print(hmm.viterbi(np.array([[0,0,0]])))
print(hmm.viterbi(np.array([[1,1,1]])))
print(hmm.viterbi(np.array([[0,1,0,1,0,1]])))