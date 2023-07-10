import sys
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class EM_algorithm_GMM():
    def __init__(self, n_clusters = 4):
        np.random.seed(seed = 42)
        self.n_clusters = n_clusters
        self.threshold = 0.00001
        self.n_feature = None
        self.n_samples = None
        self.max = None
        self.mu = None
        self.pi = None
        self.gamma = None
        self.Sigma = None
            
    def _init_params(self, X, iter_max):
        
        self.n_samples, self.n_feature = X.shape
        self.max_iter = iter_max
        self.mu = np.random.randn(self.n_clusters, self.n_feature)
        self.pi = np.ones((self.n_clusters)) * (1 / self.n_clusters)
        self.gamma = np.random.rand(self.n_samples, self.n_clusters)
        self.Sigma = np.zeros((self.n_clusters, self.n_feature, self.n_feature))
        for m in range(self.n_clusters):
            self.Sigma[m] = np.eye(self.n_feature)
    
    def _expectation(self, x):
        gamma = np.zeros_like(self.gamma)
        for n in range(self.n_samples):
            sum_pi = 0
            pi = np.zeros((self.n_clusters))
            for k in range(self.n_clusters):
                pi[k] = self.pi[k] * self._multivariate_gaussian(x[n], self.mu[k], self.Sigma[k])
                sum_pi += pi[k]
            for k in range(self.n_clusters):
                gamma[n][k] = pi[k] / sum_pi
        self.gamma = gamma

    def _maximization(self, x):
        pi = np.zeros_like(self.pi)
        mu = np.zeros_like(self.mu)
        Sigma = np.zeros_like(self.Sigma)
        for m in range(self.n_clusters):
            sum_gamma = 0
            sum_gamma_x = np.zeros_like(self.mu[m])
            sum_gamma_xx = np.zeros_like(self.Sigma[m])
            for n in range(self.n_samples):
                sum_gamma += self.gamma[n][m]
                sum_gamma_x += self.gamma[n][m] * x[n]
                sum_gamma_xx += self.gamma[n][m] * x[n].reshape((-1, 1)) @ x[n].reshape((-1, 1)).T
            pi[m] = sum_gamma / self.n_samples
            mu[m] = sum_gamma_x / sum_gamma
            Sigma[m] = sum_gamma_xx / sum_gamma - mu[m].reshape((-1, 1)) @ mu[m].reshape((-1, 1)).T
        self.pi = pi
        self.mu = mu
        self.Sigma = Sigma
    
    def _multivariate_gaussian(self, x, mu, Sigma):
        Sigma_inv = np.linalg.pinv(Sigma)
        Sigma_det = np.linalg.det(Sigma)
        const = 1 / (np.sqrt(((2 * np.pi) ** self.n_feature)) * np.sqrt(Sigma_det))
        exponent = -(x - mu).T @ Sigma_inv @ (x - mu) / 2.0
        return  const * np.exp(exponent)
    
    def _log_likelihood(self, x):
        total_likelihood = 0
        for n in range(self.n_samples):
            total = 0
            for k in range(self.n_clusters):
                total += self.pi[k] * self._multivariate_gaussian(x[n], self.mu[k], self.Sigma[k])
            total_likelihood += np.log(total)
        return total_likelihood 
    
    def fit(self, x, iter_max = 50):
        self._init_params(x, iter_max = iter_max) 
        prelikelihood = -100000
        for i in range(self.max_iter):
            self._expectation(x)
            self._maximization(x)
            likelihood = self._log_likelihood(x)
            print(f'number of iteration {i + 1} : log-likelihood {likelihood}')
            if abs(prelikelihood - likelihood) < self.threshold:
                print("Early Stop!")
                break
            else:
                prelikelihood = likelihood
            
        return self.gamma, self.pi, self.mu, self.Sigma
        
path_to_csv = './x.csv'
data = pd.read_csv(path_to_csv)
X = data.values
Y = [float(s) for s in data.columns]
X = np.vstack([Y, X])


#assuming the number of cluster is 4
n_clusters = 4
model = EM_algorithm_GMM(n_clusters=n_clusters)
gamma, pi, mu, Sigma = model.fit(X)
labels = np.argmax(gamma, axis=1)

dfz = pd.DataFrame(gamma).to_csv('z.csv', index=False, header=None)
with open('params.dat', 'w') as f:
    f.write(f'Weight: pi\n')
    f.write(str(pi))
    f.write(f'\nMeans of Gaussian Functions: mu\n')
    f.write(str(mu))
    f.write(f'\nVariances of Gaussian Functions: Sigma\n')
    f.write(str(Sigma))
    f.close()

cm = plt.get_cmap("tab10")
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
n_sample = X.shape[0]

for n in range(n_sample):
    ax.plot([X[n][0]], [X[n][1]], [X[n][2]], "o", color=cm(labels[n]))
ax.view_init(elev=30, azim=45)
plt.show()


"""
#assuming the number of cluster is 3
n_clusters = 3
model = EM_algorithm_GMM(n_clusters=n_clusters)
gamma, pi, mu, Sigma = model.fit(X)
labels = np.argmax(gamma, axis=1)

cm = plt.get_cmap("tab10")
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
n_sample = X.shape[0]

for n in range(n_sample):
    ax.plot([X[n][0]], [X[n][1]], [X[n][2]], "o", color=cm(labels[n]))
ax.view_init(elev=30, azim=45)
plt.show()
"""

"""
#assuming the number of cluster is 5
n_clusters = 5
model = EM_algorithm_GMM(n_clusters=n_clusters)
gamma, pi, mu, Sigma = model.fit(X)
labels = np.argmax(gamma, axis=1)

cm = plt.get_cmap("tab10")
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
n_sample = X.shape[0]

for n in range(n_sample):
    ax.plot([X[n][0]], [X[n][1]], [X[n][2]], "o", color=cm(labels[n]))
ax.view_init(elev=30, azim=45)
plt.show()
"""