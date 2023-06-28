import numpy as np

b = np.array([1, 2, 3, 4]).reshape((-1, 2))
c = np.array([5, 6, 7, 8]).reshape((-1, 2))
print(b * c)