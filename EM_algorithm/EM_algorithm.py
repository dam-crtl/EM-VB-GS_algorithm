import sys
import numpy as np
import pandas as pd

class EM_algorithm_GMM():
    def __init__(self, n_clusters = 4, n_dimensions = 3, n_samples = 10000):
        np.random.seed(seed=42)
        self.n_clusters = n_clusters
        self.n_dimensions = n_dimensions
        self.n_samples = n_samples
        self.max_iter = 50
        self.mu = 10 * np.random.randn(self.n_clusters, self.n_dimensions)
        self.pi = np.ones((self.n_clusters)) * (1 / self.n_clusters)
        self.gamma = np.random.rand(self.n_samples, self.n_clusters)
        self.Sigma = np.zeros((self.n_clusters, self.n_dimensions, self.n_dimensions))
        for m in range(self.n_clusters):
            self.Sigma[m] = np.eye(self.n_dimensions)
    
    def fit(self, x):
        for i in range(self.max_iter):
            self._e_step(x)
            self._m_step(x)
            print(f'nber of iteration:{i + 1}\t\t\tloss:{self._log_likelihood(x)}')
        #self._create_files()
    
    def _multivariate_gaussian(self, x, mu, Sigma):
        Sigma_inv = np.linalg.inv(Sigma)
        Sigma_det = np.linalg.det(Sigma)
        const = 1 / np.sqrt(((2 * np.pi) ** self.n_dimensions)) * np.sqrt(Sigma_det)
        exponent = -(x - mu).T @ Sigma_inv @ (x - mu) / 2.0
        return  const * np.exp(exponent)
    
    def _e_step(self, x):
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
    
    def _m_step(self, x):
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
        total_loss = 0
        for n in range(self.n_samples):
            total = 0
            for k in range(self.n_clusters):
                total += self.pi[k] * self._multivariate_gaussian(x[n], self.mu[k], self.Sigma[k])
            total_loss += np.log(total)
        return total_loss
    
    #def _create_files(self):
        #df_z = pd.DataFrame(self.gamma_list)
        #df_params = pd.DataFrame({'pi': self.pi_list, 'mu': self.mu_list, 'Column3': self.Sigma_list})
        ##df_z.to_csv("z.csv")
        ##df_params.to_csv("param.dat")
        #return
        
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