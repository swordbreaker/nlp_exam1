import numpy as np


a = np.array([
    [1,2,3,1],
    [4,5,6,1],
    [7,8,9,1],
    ])

u, s, vh = np.linalg.svd(a)

print("U:")
print(u)

print("S:")
print(s)

print("VH")
print(vh)

"""
d1 : Romeo and Juliet.
d2 : Juliet: O happy dagger!
d3 : Romeo died by dagger.
d4 : “Live free or die”, that’s the New-Hampshire’s motto.
d5 : Did you know, New-Hampshire is in New-England.

t1: romeo
t2: juliet
t3: happy
t4: dagger
t5: live
t6: die
t7: free
t8: new-hampshire
"""

a = np.array([
    [1,0,1,0,0],
    [1,1,0,0,0],
    [0,1,0,0,0],
    [0,1,1,0,0],
    [0,0,0,1,0],
    [0,0,1,1,0],
    [0,0,0,1,0],
    [0,0,0,1,1],
    ])

s, e, u = np.linalg.svd(a)
e = np.eye(e.shape[0]) * e


print(f"S{s.shape}:")
print(s)

print(f"Epsilon{e.shape}")
print(e)

print(f"U{u.shape}:")
print(u)

s2 = s[:,:2]
e2 = e[:2, :2]
u2 = u[:2,:]

print(f"S2{s2.shape}:")
print(s2)

print(f"Epsilon2{e2.shape}")
print(e2)

print(f"U2{u2.shape}:")
print(u2)

term = s2 @ e2
document = e2 @ u2

print(f"term{term.shape}")
print(term)

print(f"document{document.shape}")
print(document)

#die(5) and dagger(3)

die = term[5]
dagger = term[3]

query = (die + dagger)/2


def cos_distance(a,b):
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))


results = []
for i in range(document.shape[1]):
    cos_dist = cos_distance(document[:,i], query)
    results.append(cos_dist)

print(results)