import numpy as np

#b = np.array([1, 2, 3, 4]).reshape((-1, 1))
#b = np.array([1, 2, 3])
#c = np.array([5, 6, 7, 8]).reshape((-1, 1))
d = np.eye(3)
#ans = b.T @ d @ b
print(np.ones((4, 2, 3)) * d)
print(np.tile(d, (4, 2, 3)))