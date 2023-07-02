import sys
import numpy as np
import pandas as pd

class EM_algorithm_GMM():
    def __init__(self, n_clusters = 4):
        self.n_clusters = n_clusters
        self.n_feature = None
        self.n_samples = None
        self.max = None
        self.mu = None
        self.pi = None
        self.gamma = None
        self.Sigma = None
    return
            
    def _init_params(self, X, iter_max):
        np.random.seed(seed = 42)
        self.n_samples, self.n_feature = X.shape
        self.max_iter = iter_max
        self.mu = 10 * np.random.randn(self.n_clusters, self.n_feature)
        self.pi = np.ones((self.n_clusters)) * (1 / self.n_clusters)
        self.gamma = np.random.rand(self.n_samples, self.n_clusters)
        self.Sigma = np.zeros((self.n_clusters, self.n_feature, self.n_feature))
        for m in range(self.n_clusters):
            self.Sigma[m] = np.eye(self.n_feature)
        return
        
   
    
    def _multivariate_gaussian(self, x, mu, Sigma):
        Sigma_inv = np.linalg.inv(Sigma)
        Sigma_det = np.linalg.det(Sigma)
        const = 1 / np.sqrt(((2 * np.pi) ** self.n_feature)) * np.sqrt(Sigma_det)
        exponent = -(x - mu).T @ Sigma_inv @ (x - mu) / 2.0
        return  const * np.exp(exponent)
    
    def _expectation(self, x):
        gamma = np.zeros_like(self.gamma)
        for n in range(self.n_samples):
            pi = np.zeros((self.n_clusters))
            for k in range(self.n_clusters):
                pi[k] = self.pi[k] * self._multivariate_gaussian(x[n], self.mu[k], self.Sigma[k])
            sum_pi = np.sum(pi)
            for k in range(self.n_clusters):
                gamma[n][k] = pi[k] / sum_pi
        self.gamma = gamma
        return 
    
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
                sum_gamma_xx += self.gamma[n][m] * (x[n] - self.mu[m]).reshape((-1, 1)) @ (x[n] - self.mu[m]).reshape((-1, 1)).T
            pi[m] = sum_gamma / self.n_samples
            mu[m] = sum_gamma_x / sum_gamma
            Sigma[m] = sum_gamma_xx / sum_gamma
        self.pi = pi
        self.mu = mu
        self.Sigma = Sigma
        return
    
    def _log_likelihood(self, x):
        total_likelihood = 0
        for n in range(self.n_samples):
            total = 0
            for k in range(self.n_clusters):
                total += self.pi[k] * self._multivariate_gaussian(x[n], self.mu[k], self.Sigma[k])
            total_likelihood += np.log(total)
        return total_likelihood 
    
    def fit(self, X, iter_max):
        self._init_params(X, iter_max) 
        prelikelihood = -100000
        for i in range(self.max_iter):
            self._expectation(x)
            self._maximization(x)
            likelihood = self._log_likelihood(X)
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
#a = np.array([1, 2, 3]).reshape((-1, 1))
#matrix = a @ a.T
#print(matrix)
emgmm = EM_algorithm_GMM()
emgmm.fit(X)