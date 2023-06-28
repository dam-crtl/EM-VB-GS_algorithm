import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

ax.set_title("Helix", size=20)

ax.set_xlabel("x", size=14)
ax.set_ylabel("y", size=14)
ax.set_zlabel("z", size=14)

df = pd.read_csv("./x.csv", header=None)

x = df.iloc[:, 0]
y = df.iloc[:, 1]
z = df.iloc[:, 2]
ax.scatter(x, y, z, s = 40, color='blue')
plt.show()