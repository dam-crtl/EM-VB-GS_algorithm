import numpy as np
import pandas as pd
class GS_algorithm():
    def __init__(self, n_clusters = 4):
        self.n_clusters = n_clusters
        self.n_feature = None
        self.n_sample = None
        self.max_iter = 100
        
    def _init_prames(self, X):
        np.random.seed(seed=42)
        self.n_sample, self.n_feature = X.shape
    
    def _init_clusters(self, X):
        np.random.shuffle(X)
        return X[:k]
    
    def _compute_distance(point, cluster):
        return np.linalg.norm(point - cluster)
    
    def fit(X, k, max_iter):
        self._init_params(X, max_iter = max_iter)
        clusters = self._init_clusters(X, k)
        assignments = np.zeros(self.n_sample, dtype=int)

    # 反復処理
        for i in range(self.max_iter):
            # 各点について所属クラスターをサンプリングする
            for j in range(self.n_sample):
                point = X[j]
                distances = np.zeros(k)

            # 各クラスターとの距離を計算する
                for j in range(k):
                    cluster = clusters[j]
                    distances[j] = self._compute_distance(point, cluster)

                probabilities = np.exp(-distances)
                probabilities /= np.sum(probabilities)
                assignments[j] = np.random.choice(k, p=probabilities)

        # 各クラスターのパラメータを更新する
            for j in range(k):
                cluster_points = X[assignments == j]
                if len(cluster_points) > 0:
                    clusters[j] = np.mean(cluster_points, axis=0)
            
            

    return assignments, clusters

path_to_csv = './x.csv'
data = pd.read_csv(path_to_csv)
X = data.values
Y = [float(s) for s in data.columns]
X = np.vstack([Y, X])

# クラスタリングのパラメータを設定する
num_clusters = 4
num_iterations = 1000

# ギブスサンプリングによるクラスタリングを実行する
assignments, clusters = gibbs_sampling(X, n_clusters = 4, )

# 結果を表示する
print("クラスター割り当て:")
print(assignments)
print("クラスターの中心:")
print(clusters)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

colors = ['red', 'green', 'blue', 'orange']
cluster_colors = [colors[label] for label in assignments]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=cluster_colors)
plt.show()