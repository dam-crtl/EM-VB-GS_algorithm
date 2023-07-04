import numpy as np

#b = np.array([1, 2, 3, 4]).reshape((-1, 1))
#b = np.array([1, 2, 3])
#c = np.array([5, 6, 7, 8]).reshape((-1, 1))
e = np.eye(3)
#ans = b.T @ d @ b
#b = np.array([[1, 2, 3], [1, 2 ,3]])

a = np.array([1, 2, 3])[None, :]
b = np.array([1, 2, 3])[:, None]
c = np.array([1, 2, 3]).reshape((-1, 1))
d = np.array([1, 2, 3]).reshape((1, -1))
#pi = c / np.sum(c, keepdims=True)

print(a)
print(b)
print(c)
print(d)
print(e / np.array([1, 2, 3])[:, None])
print(e / np.array([1, 2, 3]))
#a.append(1)
#print(a)
#ans = (b - c).T @ d @ (b - c)
#print(np.sum(b, axis=0)[None, :])
#print(np.sum(b, axis=0))
#print(ans)
#print(np.ones((4, 2, 3)) * d)
e = np.eye(3)
print(np.tile(e, (4, 1, 1)))