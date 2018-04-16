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


print("S:")
print(s)

print("Epsilon")
print(e)

print("U:")
print(u)


e2 = e[:2, :2]

s2 = s[:,:2]

u2 = u[:2,:]

a2 = s2 @ e2 @ u2

term = s2 @ e2
document = e2 @ u2

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