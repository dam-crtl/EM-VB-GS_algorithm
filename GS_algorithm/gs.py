import numpy as np

def initialize_clusters(data, k):
    # データからk個のクラスターを初期化する
    # ランダムにk個のデータ点を選ぶ
    np.random.shuffle(data)
    return data[:k]

def compute_distance(point, cluster):
    # 点とクラスターの間の距離を計算する
    return np.linalg.norm(point - cluster)

def gibbs_sampling(data, k, iterations):
    # ギブスサンプリングによるクラスタリングを実行する
    num_points = data.shape[0]
    num_dimensions = data.shape[1]

    # クラスターの初期化
    clusters = initialize_clusters(data, k)

    # 各点の所属クラスターを格納する配列を初期化
    assignments = np.zeros(num_points, dtype=int)

    # 反復処理
    for _ in range(iterations):
        # 各点について所属クラスターをサンプリングする
        for i in range(num_points):
            point = data[i]
            distances = np.zeros(k)

            # 各クラスターとの距離を計算する
            for j in range(k):
                cluster = clusters[j]
                distances[j] = compute_distance(point, cluster)

            # サンプリング分布から所属クラスターを選ぶ
            probabilities = np.exp(-distances)
            probabilities /= np.sum(probabilities)
            assignments[i] = np.random.choice(k, p=probabilities)

        # 各クラスターのパラメータを更新する
        for j in range(k):
            cluster_points = data[assignments == j]
            if len(cluster_points) > 0:
                clusters[j] = np.mean(cluster_points, axis=0)

    return assignments, clusters

import pandas as pd
# データを生成する（ランダムな3次元の点）
path_to_csv = './x.csv'
data = pd.read_csv(path_to_csv)
X = data.values
Y = [float(s) for s in data.columns]
data = np.vstack([Y, X])
num_points = 10000
#data = np.random.rand(num_points, 3)

# クラスタリングのパラメータを設定する
num_clusters = 4
num_iterations = 1000

# ギブスサンプリングによるクラスタリングを実行する
assignments, clusters = gibbs_sampling(data, num_clusters, num_iterations)

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