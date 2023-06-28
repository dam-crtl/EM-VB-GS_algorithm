import numpy as np
import pandas as pd
# EMアルゴリズムの実装

def multivariate_gaussian(x, mean, cov):
    """
    多変量ガウス分布の確率密度関数を計算する関数
    x: データ点（1次元または2次元配列）
    mean: 平均ベクトル
    cov: 共分散行列
    """
    d = len(mean)  # 次元数
    coeff = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(cov) ** 0.5)
    x_minus_mean = x - mean
    exponent = -0.5 * np.dot(np.dot(x_minus_mean, np.linalg.inv(cov)), x_minus_mean.T)
    return coeff * np.exp(exponent)

def initialize_parameters(data, n_clusters):
    """
    パラメータ（平均、共分散行列、混合係数）の初期化
    data: 入力データ
    n_clusters: クラスター数
    """
    n_samples, n_features = data.shape

    # 平均の初期化
    means = np.zeros((n_clusters, n_features))
    for i in range(n_clusters):
        means[i] = data[np.random.choice(n_samples)]

    # 共分散行列の初期化
    covs = np.zeros((n_clusters, n_features, n_features))
    for i in range(n_clusters):
        covs[i] = np.eye(n_features)

    # 混合係数の初期化
    weights = np.ones(n_clusters) / n_clusters

    return means, covs, weights

def expectation(data, means, covs, weights):
    """
    Eステップ：データ点の所属クラスターの事後確率を計算する
    data: 入力データ
    means: 各クラスターの平均ベクトル
    covs: 各クラスターの共分散行列
    weights: 各クラスターの混合係数
    """
    n_samples = data.shape[0]
    n_clusters = len(means)
    posteriors = np.zeros((n_samples, n_clusters))

    for i in range(n_samples):
        for j in range(n_clusters):
            posteriors[i, j] = weights[j] * multivariate_gaussian(data[i], means[j], covs[j])

        # 正規化
        posteriors[i] /= np.sum(posteriors[i])

    return posteriors

def maximization(data, posteriors):
    """
    Mステップ：パラメータ（平均、共分散行列、混合係数）を更新する
    data: 入力データ
    posteriors: データ点の所属クラスターの事後確率
    """
    n_samples, n_features = data.shape
    n_clusters = posteriors.shape[1]

    # 平均の更新
    means = np.zeros((n_clusters, n_features))
    for j in range(n_clusters):
        means[j] = np.average(data, axis=0, weights=posteriors[:, j])

    # 共分散行列の更新
    covs = np.zeros((n_clusters, n_features, n_features))
    for j in range(n_clusters):
        diff = data - means[j]
        weighted_diff = np.dot(posteriors[:, j] * diff.T, diff)
        covs[j] = weighted_diff / np.sum(posteriors[:, j])

    # 混合係数の更新
    weights = np.mean(posteriors, axis=0)

    return means, covs, weights

def gmm(data, n_clusters, n_iterations):
    """
    GMMを実行する関数
    data: 入力データ
    n_clusters: クラスター数
    n_iterations: 反復回数
    """
    means, covs, weights = initialize_parameters(data, n_clusters)

    for _ in range(n_iterations):
        posteriors = expectation(data, means, covs, weights)
        means, covs, weights = maximization(data, posteriors)

    return means, covs, weights

# データの生成
np.random.seed(42)

# クラスターごとのデータ点の数
n_samples = 100

# クラスター1
mean1 = np.array([0, 0, 0])
cov1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
cluster1 = np.random.multivariate_normal(mean1, cov1, n_samples)

# クラスター2
mean2 = np.array([5, 5, 5])
cov2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
cluster2 = np.random.multivariate_normal(mean2, cov2, n_samples)

# クラスター3
mean3 = np.array([-5, -5, -5])
cov3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
cluster3 = np.random.multivariate_normal(mean3, cov3, n_samples)

# クラスター4
mean4 = np.array([5, -5, 0])
cov4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
cluster4 = np.random.multivariate_normal(mean4, cov4, n_samples)

# データ行列の作成
data = np.concatenate([cluster1, cluster2, cluster3, cluster4])

path_to_csv = './x.csv'
data = pd.read_csv(path_to_csv)
X = data.values
Y = [float(s) for s in data.columns]
X = np.vstack([Y, X])

# GMMの実行
n_clusters = 4
n_iterations = 100
means, covs, weights = gmm(X, n_clusters, n_iterations)

# 各データ点の所属クラスターを予測
posteriors = expectation(X, means, covs, weights)
predicted_labels = np.argmax(posteriors, axis=1)

# クラスターごとに色を設定
colors = ['red', 'green', 'blue', 'orange']
cluster_colors = [colors[label] for label in predicted_labels]

# データ点とクラスターをプロット
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_colors)
plt.show()
