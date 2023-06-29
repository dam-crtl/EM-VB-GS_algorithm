import numpy as np
from scipy.stats import multivariate_normal

class BayesianGMM:
    def __init__(self, n_clusters, alpha_0, beta_0, m_0, W_0):
        self.n_clusters = n_clusters
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.m_0 = m_0
        self.W_0 = W_0
        self.cluster_params = []

    def _initialize_cluster_params(self, data):
        n_samples, n_features = data.shape
        for _ in range(self.n_clusters):
            mean = np.random.randn(n_features)
            precision = np.linalg.inv(self.W_0)
            self.cluster_params.append({'mean': mean, 'precision': precision})

    def _update_cluster_params(self, data, responsibilities):
        n_samples = data.shape[0]
        for j in range(self.n_clusters):
            N_j = np.sum(responsibilities[:, j])

            # パラメータの更新式
            beta_n = self.beta_0 + N_j
            alpha_n = self.alpha_0 + N_j
            m_n = (self.beta_0 * self.m_0 + np.sum(responsibilities[:, j] * data.T, axis=1)) / beta_n
            W_n_inv = np.linalg.inv(self.W_0) + np.sum(responsibilities[:, j, np.newaxis, np.newaxis] *
                np.array([np.outer(x - self.m_0, x - self.m_0) for x in data]), axis=0
            ) + (self.beta_0 * N_j / beta_n) * np.outer(self.m_0 - self.cluster_params[j]['mean'], self.m_0 - self.cluster_params[j]['mean'])
            W_n = np.linalg.inv(W_n_inv)

            self.cluster_params[j] = {'mean': m_n, 'precision': W_n}

    def _update_responsibilities(self, data):
        n_samples = data.shape[0]
        responsibilities = np.zeros((n_samples, self.n_clusters))

        for i in range(n_samples):
            for j in range(self.n_clusters):
                responsibilities[i, j] = multivariate_normal.pdf(data[i], self.cluster_params[j]['mean'], np.linalg.inv(self.cluster_params[j]['precision']))

        #responsibilities = responsibilities * np.repeat(self.alpha_0 / self.n_clusters, n_samples).reshape(-1, self.n_clusters)
        responsibilities = responsibilities * self.alpha_0 / self.n_clusters
        responsibilities = responsibilities / np.sum(responsibilities, axis=1)[:, np.newaxis]

        return responsibilities

    def fit(self, data, max_iterations=10):
        self._initialize_cluster_params(data)

        for _ in range(max_iterations):
            responsibilities = self._update_responsibilities(data)
            self._update_cluster_params(data, responsibilities)

    def predict(self, data):
        responsibilities = self._update_responsibilities(data)
        print(responsibilities)
        return np.argmax(responsibilities, axis=1)

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

# 変分ベイズアルゴリズムの実行
n_clusters = 4
alpha_0 = 1.0
beta_0 = 1.0
m_0 = np.zeros(3)
W_0 = np.eye(3)
bayesian_gmm = BayesianGMM(n_clusters, alpha_0, beta_0, m_0, W_0)
bayesian_gmm.fit(data)

# 各データ点の所属クラスターを予測
predicted_labels = bayesian_gmm.predict(data)
print(predicted_labels)
# クラスターごとに色を設定
colors = ['red', 'green', 'blue', 'orange']
cluster_colors = [colors[label] for label in predicted_labels]

# データ点とクラスターをプロット
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=cluster_colors)
plt.show()
