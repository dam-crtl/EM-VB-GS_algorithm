import sys
import numpy as np
import pandas as pd
from scipy.special import digamma, logsumexp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

class VB_algorithm_GMM:
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
        self.gamma = None
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
        self.gamma = np.ones((self.n_sample, self.n_clusters)) * (self.n_sample / self.n_clusters)
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

    def _expectation(self, X):
        pi = digamma(self.alpha) - digamma(np.sum(self.alpha))
        log_Lambda = np.zeros(self.n_clusters)
        for k in range(self.n_clusters):
            sum_digamma = 0
            for i in range(self.n_feature):
                sum_digamma += digamma((self.nu[k] + 1 - i) / 2)
            log_Lambda[k] = np.array([sum_digamma])[0][0] + self.n_feature * np.log(2) + np.log(np.linalg.det(self.W[k]))
        rho = np.zeros((self.n_sample, self.n_clusters))
        for n in range(self.n_sample):
            for k in range(self.n_clusters):
                residual = X[n] - self.m[k]
                rho[n][k] = (pi[k] + log_Lambda[k] / 2.0 - (self.n_feature / (2 * self.beta[k]))
                             - (self.nu[k] / 2.0) * (residual.T @ self.W[k] @ residual))[0]
        log_gamma = rho - logsumexp(rho, axis=1).reshape(-1, 1)
        self.gamma = np.exp(log_gamma)

    def _maximization(self, X):
        sum_gamma = np.zeros(self.n_clusters)
        sum_gamma_x = np.zeros((self.n_clusters, self.n_feature))
        sum_gamma_xx = np.zeros((self.n_clusters, self.n_feature, self.n_feature))
        for k in range(self.n_clusters):
            for n in range(self.n_sample):
                sum_gamma[k] += self.gamma[n][k]
                sum_gamma_x[k] += self.gamma[n][k] * X[n]
                sum_gamma_xx[k] += self.gamma[n][k] * X[n].reshape((-1, 1)) @ X[n].reshape((-1, 1)).T
        self.alpha = self.alpha0 + sum_gamma
        self.beta = self.beta0 + sum_gamma
        for k in range(self.n_clusters):
            self.m[k] = (self.beta0 * self.m0 + sum_gamma_x[k]) / (self.beta0 + sum_gamma[k])
        self.mu = self.m
        for k in range(self.n_clusters):
            self.W[k] = np.linalg.inv(np.linalg.inv(self.W0) + self.beta0 * self.m0.reshape((-1, 1)) @ self.m0.reshape((-1, 1)).T 
                                      + sum_gamma_xx[k] - self.beta[k] * self.m[k].reshape((-1, 1)) @ self.m[k].reshape((-1, 1)).T)
            self.nu[k] = self.nu0 + sum_gamma[k]
        for k in range(self.n_clusters):
            self.Sigma[k] = np.linalg.inv(self.nu[k] * self.W[k])
        self.pi = self.alpha / np.sum(self.alpha)

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
                
        return self.gamma, self.pi, self.mu, self.Sigma

    
if __name__== '__main__':
    
    path_to_csv = './' + sys.argv[1]
    data = pd.read_csv(path_to_csv)
    X = data.values
    Y = [float(s) for s in data.columns]
    X = np.vstack([Y, X])
    
    """
    #assuming the number of cluster is 4
    n_clusters = 4
    model = VB_algorithm_GMM(n_clusters=n_clusters)
    gamma, pi, mu, Sigma = model.fit(X)
    labels = np.argmax(gamma, axis=1)

    dfz = pd.DataFrame(gamma).to_csv(sys.argv[2], index=False, header=None)
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
    
    """
    #assuming the number of cluster is 3
    n_clusters = 3
    model = VB_algorithm_GMM(n_clusters=n_clusters)
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

    
    #assuming the number of cluster is 5
    n_clusters = 5
    model = VB_algorithm_GMM(n_clusters=n_clusters)
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
    