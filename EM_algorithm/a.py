import numpy as np
import pandas as pd

def multivariate_gaussian(x, mean, cov):
    d = len(mean)
    coeff = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(cov) ** 0.5)
    x_minus_mean = x - mean
    exponent = -0.5 * np.dot(np.dot(x_minus_mean, np.linalg.inv(cov)), x_minus_mean.T)
    return coeff * np.exp(exponent)

def initialize_parameters(data, n_clusters):
    n_samples, n_features = data.shape
    np.random.seed(seed=42)
    means = np.zeros((n_clusters, n_features))
    for i in range(n_clusters):
        means[i] = data[np.random.choice(n_samples)]

    covs = np.zeros((n_clusters, n_features, n_features))
    for i in range(n_clusters):
        covs[i] = np.eye(n_features)

    pi = np.ones(n_clusters) / n_clusters

    return means, covs, pi

def expectation(data, means, covs, pi):

    n_samples = data.shape[0]
    n_clusters = len(means)
    posteriors = np.zeros((n_samples, n_clusters))

    for i in range(n_samples):
        for j in range(n_clusters):
            posteriors[i, j] = pi[j] * multivariate_gaussian(data[i], means[j], covs[j])

        posteriors[i] /= np.sum(posteriors[i])

    return posteriors

def maximization(data, posteriors):

    n_samples, n_features = data.shape
    n_clusters = posteriors.shape[1]

    means = np.zeros((n_clusters, n_features))
    for j in range(n_clusters):
        means[j] = np.average(data, axis=0, weights=posteriors[:, j])

    covs = np.zeros((n_clusters, n_features, n_features))
    for j in range(n_clusters):
        diff = data - means[j]
        weighted_diff = np.dot(posteriors[:, j] * diff.T, diff)
        covs[j] = weighted_diff / np.sum(posteriors[:, j])

    # 混合係数の更新
    pi = np.mean(posteriors, axis=0)

    return means, covs, pi

def gmm(data, n_clusters, n_iterations):
    means, covs, pi = initialize_parameters(data, n_clusters)

    for _ in range(n_iterations):
        posteriors = expectation(data, means, covs, pi)
        means, covs, pi = maximization(data, posteriors)

    return means, covs, pi

path_to_csv = './x.csv'
data = pd.read_csv(path_to_csv)
X = data.values
Y = [float(s) for s in data.columns]
X = np.vstack([Y, X])

n_clusters = 4
n_iterations = 100
means, covs, pi = gmm(X, n_clusters, n_iterations)
posteriors = expectation(X, means, covs, pi)

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
