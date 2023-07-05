import numpy as np

a = np.array([[1, 2], [3, 4]])
#b = a.tolist()
#print(b)
#str_a = map(str, a.tolist())
#print(str_a[0])
f = open('myfile.txt', 'w')
f.write(str(a))
f.close()
