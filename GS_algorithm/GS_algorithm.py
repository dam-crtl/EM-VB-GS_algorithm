import sys
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, wishart, dirichlet
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

class GS_algorithm_GMM:
    def __init__(self, n_clusters=4):
        np.random.seed(seed=42)
        self.n_clusters = n_clusters
        self.alpha0 = 0.01
        self.beta0 = 1.0
        self.threshold = 0.01
        self.n_sample = None
        self.n_feature = None
        self.m0 = None
        self.W0 = None
        self.nu0 = None
        self.z = None
        self.alpha = None
        self.beta = None
        self.m = None
        self.W = None
        self.pi = None
        self.nu = None
        self.Sigma = None

    def _init_params(self, X):
        self.n_sample, self.n_feature = X.shape
        self.m0 = np.random.randn(self.n_feature)
        self.W0 = np.eye(self.n_feature)
        self.nu0 = np.array([self.n_feature])
        self.z = np.ones((self.n_sample, self.n_clusters)) * (self.n_sample / self.n_clusters)
        self.z_prob = np.ones((self.n_sample, self.n_clusters)) * (self.n_sample / self.n_clusters)
        self.alpha = np.ones(self.n_clusters) * self.alpha0
        self.beta = np.ones(self.n_clusters) * self.beta0
        self.m = np.random.randn(self.n_clusters, self.n_feature)
        self.pi = np.ones((self.n_clusters)) / self.n_clusters
        self.nu = np.ones((self.n_clusters, 1)) * self.n_feature
        self.W = np.zeros((self.n_clusters, self.n_feature, self.n_feature))
        self.Sigma = np.zeros((self.n_clusters, self.n_feature, self.n_feature))
        for k in range(self.n_clusters):
            self.W[k] = np.eye(self.n_feature)
            self.Sigma[k] = np.linalg.inv(self.nu[k] * self.W[k])
        self.mu = np.random.randn(self.n_clusters, self.n_feature)

    def _expectation(self, x):
        z_prob = np.zeros_like(self.z_prob)
        for n in range(self.n_sample):
            sum_pi = 0
            pi = np.zeros((self.n_clusters))
            for k in range(self.n_clusters):
                pi[k] = self.pi[k] * self._multivariate_gaussian(x[n], self.mu[k], self.Sigma[k])
                sum_pi += pi[k]
            for k in range(self.n_clusters):
                z_prob[n][k] = pi[k] / sum_pi
        self.z_prob = z_prob

    def _maximization(self, X):
        for n in range(self.n_sample):
            self.z[n] = np.random.multinomial(n=1, pvals=self.z_prob[n]).flatten()

        sum_z = np.zeros(self.n_clusters)
        sum_z_x = np.zeros((self.n_clusters, self.n_feature))
        sum_z_xx = np.zeros((self.n_clusters, self.n_feature, self.n_feature))
        for k in range(self.n_clusters):
            for n in range(self.n_sample):
                sum_z[k] += self.z[n][k]
                sum_z_x[k] += self.z[n][k] * X[n]
                sum_z_xx[k] += self.z[n][k] * X[n].reshape((-1, 1)) @ X[n].reshape((-1, 1)).T
        self.alpha = self.alpha0 + sum_z
        self.beta = self.beta0 + sum_z
        for k in range(self.n_clusters):
            self.m[k] = (self.beta0 * self.m0 + sum_z_x[k]) / (self.beta0 + sum_z[k])
        for k in range(self.n_clusters):
            self.W[k] = np.linalg.inv(np.linalg.inv(self.W0) + self.beta0 * self.m0.reshape((-1, 1)) @ self.m0.reshape((-1, 1)).T 
                                      + sum_z_xx[k] - self.beta[k] * self.m[k].reshape((-1, 1)) @ self.m[k].reshape((-1, 1)).T)
            self.nu[k] = self.nu0 + sum_z[k]
        for k in range(self.n_clusters):
            self.mu[k] = np.random.multivariate_normal(mean=self.m[k], cov=self.Sigma[k]).flatten()
            self.Sigma[k] = np.linalg.inv(wishart.rvs(size=1, df=self.nu[k][0], scale=self.W[k])).reshape((self.n_feature, self.n_feature))
        self.pi = dirichlet.rvs(size=1, alpha=self.alpha).flatten()

    def _multivariate_gaussian(self, x, mu, Sigma):
        Sigma_inv = np.linalg.inv(Sigma)
        Sigma_det = np.linalg.det(Sigma)
        exponent = -(x - mu).T @ Sigma_inv @ (x - mu) / 2.0
        const = 1 / ((np.sqrt(2 * np.pi) ** self.n_feature) * np.sqrt(Sigma_det))
        return const * np.exp(exponent)

    def _mixed_gaussian(self, X, mu, Sigma, pi):
        output = np.zeros((self.n_sample))
        for i in range(self.n_clusters):
            for j in range(self.n_sample):
                output[j] += pi[i] * self._multivariate_gaussian(X[j], mu[i], Sigma[i])
        return output

    def _log_likelihood(self, X):
        gaussians = self._mixed_gaussian(X, self.mu, self.Sigma, self.pi)
        total_likelihood = 0
        for i in range(self.n_sample):
            total_likelihood += np.log(gaussians[i])
            
        return total_likelihood

    def fit(self, X, iter_max = 50):
        self._init_params(X)
        prelikelihood = -1000000
        for i in range(iter_max):
            self._expectation(X)
            self._maximization(X)
            likelihood = self._log_likelihood(X)
            print(f"number of iteration {i + 1} : log-likelihood {likelihood}")
            if np.abs(likelihood - prelikelihood) < self.threshold:
                print("Early Stop!")
                break
            else:
                prelikelihood = likelihood
                
        return self.z, self.pi, self.mu, self.Sigma

if __name__== '__main__':
    
    path_to_csv = './' + sys.argv[1]
    data = pd.read_csv(path_to_csv)
    X = data.values
    Y = [float(s) for s in data.columns]
    X = np.vstack([Y, X])
    
    
    #assuming the number of cluster is 4
    n_clusters = 4
    model = GS_algorithm_GMM(n_clusters=n_clusters)
    z, pi, mu, Sigma = model.fit(X)
    labels = np.argmax(z, axis=1)

    dfz = pd.DataFrame(z).to_csv(sys.argv[2], index=False, header=None)
    with open(sys.argv[3], 'w') as f:
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
    model = GS_algorithm_GMM(n_clusters=n_clusters)
    z, pi, mu, Sigma = model.fit(X)
    labels = np.argmax(z, axis=1)

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
    model = GS_algorithm_GMM(n_clusters=n_clusters)
    z, pi, mu, Sigma = model.fit(X)
    labels = np.argmax(z, axis=1)

    cm = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    n_sample = X.shape[0]

    for n in range(n_sample):
        ax.plot([X[n][0]], [X[n][1]], [X[n][2]], "o", color=cm(labels[n]))
    ax.view_init(elev=30, azim=45)
    plt.show()
    """