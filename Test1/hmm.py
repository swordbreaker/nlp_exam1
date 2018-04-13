import numpy as np
from graphviz import Digraph

# Der Vorwärts-Algoritmus berechnet rekursiv die Wahrscheinlichkeiten alpha_t(i)=fw[t,i]=P(o_1...o_t|x_t=i), 
# d.h. die Wahrscheinlichkeit nach t Schritten die Beobachtungssequenz o_1...o_t (gemäss der Vorgabe) gemacht
# zu haben und aktuell im (inneren) Zustand x_i zu sein. Dabei werden rekursive pro Zeitschritt (beginnend mit
# dem Startzustand p0), pro Iteration die Wahrscheinlichkeiten (für alle Zustände) für einen nächsten Zeitschritt
# berechnet. Die gesuchte (totale) Wahrscheinlichkeit P(o) für die gegebene Beobachtung o ist schlussendlich die 
# Summe der Wahrscheinlichkeiten nach dem letzten Zeitschritt (Summe der Einträge der letzten Spalte)!
# Die hier implementierte Funktion berechnet aus den Angaben:
# U    (Übergangswahrscheinlichkeiten)
# E    (Emissionswahrscheinlichkeiten)
# p0   (Startwahrscheinlichkeiten)
# o    (Beobachtung)
# die gesuchte (Beobachtungs-)Wahrscheinlichkeit und die Matrix fw mit den einzelnen Wahrscheinlichkeiten!
def hmm_forward(U,E,p0,o):
    # Anzahl Beobachtungen bestimmen:
    k = o.size
    # Anzahl innere Zustände bestimmen:
    (n,m) = E.shape
    # Matrix mit den einzelnen Wahrscheinlichkeiten in der richtigen Grösse (Zeilenzahl = Anzahl Zustände,
    # Spaltenzahl = Anzahl Beobachtungen) initielisieren:
    fw = np.zeros( (n,k) )
    # Wahrscheinlichkeiten für die erste Beobachtung berechnen und in der ersten Spalte speichern:
    fw[:, 0] = p0*E[:,int(o[0,0])]
    # Für die weiteren Beobachtungen:
    for obs_ind in range(1,k,1):
        # Die Wahrscheinlichkeiten für die letzte Beobachtung aus der Matrix fw lesen (letzte geschriebene Spalte).
        # Für die weitere Berechnung wird die Spalte noch in einen Zeilenvektor umgewandelt.
        fw_old = np.transpose(fw[:,obs_ind-1])
        # Berechnung der neuen Wahrscheinlichkeiten (ergibt die neue Spalte in der Matrix fw). Dabei werden die alten 
        # Wahrscheinlichkeiten mit der Übertragungsmatrix multipliziert (Matrixmultiplikation) (ergibt die Wahr-
        # scheinlichkeiten nach t Zeitschritten in einem bestimmten Zustand zu sein)! Nun werden diese Werte noch
        # mit den entsprechenden Beobachtungswahrscheinlichkeiten multipliziert:
        fw[:, obs_ind] = fw_old.dot(U)*E[:,int(o[0,obs_ind])]
    # Nachdem alle Zeitschritte abgearbeitet sind kann die Beobachtungswahrscheinlichkeit P(o) als Summe der Werte
    # in der letzten Spalte berechnet werden:
    p = np.sum(fw[:,k-1])    
    # Rückgabe der gesuchten Wahrscheinlichkeit (p) und der berechneten Zwischenresultate (fw)
    return p,fw

# Der Rückwärts-Algoritmus berechnet rekursiv die Wahrscheinlichkeiten beta_t(i)=bw[t,i]=P(o_t+1...o_n|x_t=i), 
# d.h. die Wahrscheinlichkeit aus dem Zustand x_t=i in den verbleibenden Schritten die Beobachtungssequenz 
# o_t+1...o_n (gemäss der Vorgabe) zu machen. Dabei werden rekursive pro Zeitschritt (beginnend mit
# dem Endzustand), pro Iteration die Wahrscheinlichkeiten (für alle Zustände) rückwärts berechnet. Die gesuchte 
# (totale) Wahrscheinlichkeit P(o) für die gegebene Beobachtung o ist schlussendlich die Summe der Produkte aus
# den Wahrscheinlichkeiten der ersten Spalte mit den Beobachtungswahrscheinlichkeiten und den Startwahrscheinlichkeiten!
# Die hier implementierte Funktion berechnet aus den Angaben:
# U    (Übergangswahrscheinlichkeiten)
# E    (Emissionswahrscheinlichkeiten)
# p0   (Startwahrscheinlichkeiten)
# o    (Beobachtung)
# die gesuchte (Beobachtungs-)Wahrscheinlichkeit und die Matrix bw mit den einzelnen Wahrscheinlichkeiten!
def hmm_backward(U,E,p0,o):
    # Anzahl Beobachtungen bestimmen:
    k = o.size
    # Anzahl innere Zustände bestimmen:
    (n,m) = E.shape
    # Matrix mit den einzelnen Wahrscheinlichkeiten in der richtigen Grösse (Zeilenzahl = Anzahl Zustände,
    # Spaltenzahl = Anzahl Beobachtungen) initielisieren:
    bw = np.zeros( (n,k) )
    # Wahrscheinlichkeiten für den letzten Zeitschritt berechnen und in der letzten Spalte speichern:
    bw[:, k-1] = np.ones((n,1))[:,0]
    # Iteration:
    for obs_ind in range(1,k,1):
        # Die Wahrscheinlichkeiten für die letzten Beobachtungen aus der Matrix bw lesen (letzte geschriebene Spalte).
        bw_old = bw[:,k-obs_ind]
        # Berechnung der neuen Wahrscheinlichkeiten (ergibt die vorangehende Spalte in der Matrix bw). Dabei werden die 
        # Produkte aus Beobachtungswahrscheinlichkeiten und der alten Wahrscheinlichkeiten mit der Übertragungsmatrix
        # (von RECHTS!) multipliziert (Matrixmultiplikation)!
        bw[:, k-obs_ind-1] = U.dot(bw_old*E[:,int(o[0,k-obs_ind])])
    # Nachdem alle Zeitschritte abgearbeitet sind kann die Beobachtungswahrscheinlichkeit P(o) als Summe der Produkte
    # aus Startwahrscheinlichkeit mit den Produkten der ersten Spalte mit den Emissionswahrscheinlichkeiten berechnet werden:
    p = p0.dot(bw[:,0]*E[:,int(o[0,0])])    
    # Rückgabe der gesuchten Wahrscheinlichkeit (p) und der berechneten Zwischenresultate (bw)
    return p,bw

def hmm_forward_backward(U,E,p0,o):
    # Anzahl Beobachtungen bestimmen:
    k = o.size
    # Anzahl innere Zustände bestimmen:
    (n,m) = E.shape
    # Matrizen fw und bw mit den einzelnen Wahrscheinlichkeiten in der richtigen Grösse (Zeilenzahl = Anzahl Zustände,
    # Spaltenzahl = Anzahl Beobachtungen) initielisieren:
    fw = np.zeros( (n,k) )
    bw = np.zeros( (n,k) )
    # Wahrscheinlichkeiten für die erste Beobachtung berechnen und in der ersten Spalte speichern:
    fw[:, 0] = p0*E[:,int(o[0,0])]
    # Wahrscheinlichkeiten für den letzten Zeitschritt berechnen und in der letzten Spalte speichern:
    bw[:, k-1] = np.ones((n,1))[:,0]
    # Iteration:
    for obs_ind in range(1,k,1):
        # Die Wahrscheinlichkeiten für die letzte Beobachtung aus der Matrix fw lesen (letzte geschriebene Spalte).
        # Für die weitere Berechnung wird die Spalte noch in einen Zeilenvektor umgewandelt.
        fw_old = np.transpose(fw[:,obs_ind-1])
        # Berechnung der neuen Wahrscheinlichkeiten (ergibt die neue Spalte in der Matrix fw). Dabei werden die alten 
        # Wahrscheinlichkeiten mit der Übertragungsmatrix multipliziert (Matrixmultiplikation) (ergibt die Wahr-
        # scheinlichkeiten nach t Zeitschritten in einem bestimmten Zustand zu sein)! Nun werden diese Werte noch
        # mit den entsprechenden Beobachtungswahrscheinlichkeiten multipliziert:
        fw[:, obs_ind] = fw_old.dot(U)*E[:,int(o[0,obs_ind])]
        # Die Wahrscheinlichkeiten für die letzten Beobachtungen aus der Matrix bw lesen (letzte geschriebene Spalte).
        bw_old = bw[:,k-obs_ind]
        # Berechnung der neuen Wahrscheinlichkeiten (ergibt die vorangehende Spalte in der Matrix bw). Dabei werden die 
        # Produkte aus Beobachtungswahrscheinlichkeiten und der alten Wahrscheinlichkeiten mit der Übertragungsmatrix
        # (von RECHTS!) multipliziert (Matrixmultiplikation)!
        bw[:, k-obs_ind-1] = U.dot(bw_old*E[:,int(o[0,k-obs_ind])])
    # Nachdem alle Zeitschritte abgearbeitet sind kann die Beobachtungswahrscheinlichkeit P(o) als Summe der Werte
    # in der letzten Spalte berechnet werden:
    p = np.sum(fw[:,k-1])
    # Rückgabe der gesuchten Wahrscheinlichkeit (p) und der berechneten Zwischenresultate (fw)
    return p,fw,bw

# Der Viterbi-Algoritmus berechnet zu einer gegebenen Beobachtungssequenz o=o_1o_2...o_T die beste Sequenz der inneren
# Zustände (beste Pfadsequenz)!
# Die hier implementierte Funktion berechnet aus den folgenden Angaben:
# U    (Übergangswahrscheinlichkeiten)
# E    (Emissionswahrscheinlichkeiten)
# p0   (Startwahrscheinlichkeiten)
# o    (Beobachtung)
# die beste Pfadsequenz q und die Wahrscheinlichkeit über diese Pfadsequenz die gegebene Beobachtung zu machen.
def hmm_viterbi(U,E,p0,o):
    # Anzahl Beobachtungen bestimmen:
    k = o.size
    # Anzahl innere Zustände bestimmen:
    (n,m) = E.shape
    # Der Algorithmus berechnet die Matrix theta. Das Elemente theta[i,j] speichert die maximale Wahrscheinlichkeit nach
    # j Schritten die Beobachtungssequenz o_1...o_j gemacht zu haben und im Zustand X_i zu sein (d.h. Wenn die beste Pfad-
    # sequenz beim j-ten Schritt durch den Zustand X_i führt, so ist die Wahrscheinlichkeit gleich dm gespeicherten Wert
    # theta[i,j]). 
    # Die Matrix theta in der richtigen Grösse (Zeilenzahl = Anzahl Zustände, Spaltenzahl = Anzahl Beobachtungen)
    # initielisieren:
    theta = np.zeros( (n,k) )
    # In einer zweiten Matrix psi wird die beste Pfadsequenz gespeichert. Das Element psi[i,j] speichert den Zustand X_i-1, 
    # dies ist der Zustand auf der besten Pfadsequenz welche nach j Schritten durch den Zustand X_i führt!
    # Die Matrix psi in der richtigen Grösse (Zeilenzahl = Anzahl Zustände, Spaltenzahl = Anzahl Beobachtungen)
    # initielisieren:
    psi = np.zeros( (n,k) )
    # Wahrscheinlichkeiten für die erste Beobachtung berechnen und in der ersten Spalte von theta speichern:
    theta[:, 0] = p0*E[:,int(o[0,0])]
    # In der ersten Spalte der Matrix psi sind alle Zustände möglich:
    psi[:,0] = np.arange(n)
    # Für die weiteren Beobachtungen:
    for obs_ind in range(1,k,1):
        # Die (maximalen) Wahrscheinlichkeiten für die letzte Beobachtung aus der Matrix theta lesen (letzte geschriebene 
        # Spalte):
        theta_old = theta[:,obs_ind-1]
        # Berechnung der neuen (maximalen) Wahrscheinlichkeiten (ergibt die neue Spalte in der Matrix theta). Dabei werden
        # die alten Wahrscheinlichkeiten mit der Übertragungsmatrix multipliziert und diese Werte noch mit den entsprechenden 
        # Beobachtungswahrscheinlichkeiten multipliziert. Dies gibt für jeden Vorgängerzustand eine Wahrscheinlichkeit. Es wird
        # nun die maximale Wahrscheinlichkeit in der Matrix theta gespeichert. Zudem wird der Vorgängerzustand, welcher zu
        # dieser maximalen Wahrscheinlichkeit führt in der Matrix psi gespeichert:
        for i in range(0,n,1):
            temp = U[:,i]*theta_old
            theta[i, obs_ind] = E[i,int(o[0,obs_ind])]*max(temp)
            psi[i, obs_ind] = np.argmax(temp)
    # Nachdem alle Zeitschritte abgearbeitet sind kann die Wahrscheinlichkeit für die gemachte Beobachtung bei Durchgang über
    # die beste Pfadsequenz als das Maximum aus der letzten Spalte der Matrix theta herausgelesen werden:
    p = max(theta[:,k-1])
    # Und nun noch die beste Pfadsequenz bestimmen!
    q = np.zeros(k)
    # Beginnend beim lezten Zustand (Zeile mit maximalem Wert in der letzten Spalte der Matrix theta)
    q[obs_ind] = np.argmax(theta[:,k-1])
    # wird die beste Pfadsequenz ermittelt, indem schrittweise durch die Spalten der Matrix psi iterriert wird:
    for obs_ind in range(k-1,1,-1):
        q[obs_ind-1]=psi[int(q[obs_ind]),obs_ind]
    # Rückgabe der  besten Pfadsequenz (q) und der Wahrscheinlichkeit (p) die gemachte Beobachtung über diese Pfad-
    # sequenz gemacht zu haben:
    return p,q

class Hmm(object):
    """description of class"""

    def __init__(self, U,E,p0,o):
        """
        U    (Übergangswahrscheinlichkeiten)
        E    (Emissionswahrscheinlichkeiten)
        p0   (Startwahrscheinlichkeiten)
        o    (Beobachtung)
        die gesuchte (Beobachtungs-)Wahrscheinlichkeit und die Matrix fw mit den einzelnen Wahrscheinlichkeiten!
        """
        self.U = U
        self.E = E
        self.p0 = p0
        self.o = o

    def draw_graph(self):
        n = self.p0.shape[1]
        dot = Digraph()
        for i in range(n):
            dot.node(str(i), str(i))
            for k in range(n):
                if(self.U[i,k] != 0):
                    dot.edge(str(i), str(k), label=f"{self.U[i,k]:2.2f}")

        
        dot.render('graphviz/hmm.gv', view=True)

    def forward(self):
        return hmm_forward(self.U, self.E, self.p0, self.o)

    def backward(self):
        return hmm_backward(self.U, self.E, self.p0, self.o)

    def forward_backward(self):
        return hmm_forward_backward(self.U, self.E, self.p0, self.o)

    def viterbi(self):
        return hmm_viterbi(self.U, self.E, self.p0, self.o)



def example():
    U = np.array([
        [0.1,0.2,0.7],
        [0.2,0.7,0.1],
        [0.7,0.1,0.2]
        ])
    E = np.array([
        [0.9,0.1],
        [0.1,0.9],
        [0.0,1.0]
        ])
    p_0 = np.array([[1./3.,1./3.,1./3.]])
    o = np.array([[0.,1.,0.]])

    hmm = Hmm(U, E, p_0, o)
    p,_= hmm.forward()
    print(p)

    p,q = hmm.viterbi()

    print(f"Die beste Pfadsequenz lautet: {p}")
    print(f"Die Wahrscheinlichkeit für die Beobachtung über die beste Pfadsequenz zu machen: {q}")

    hmm.draw_graph()

example()