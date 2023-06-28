import sys
import numpy as np
import pandas as pd

class EM_algorithm_GMM():
    def __init__(self, num_clusters = 4, num_dimensions = 3, num_samples = 10000):
        np.random.seed(seed=0)
        self.num_clusters = num_clusters
        self.num_dimensions = num_dimensions
        self.num_samples = num_samples
        self.max_iter = 1
        self.mu = 3 * np.random.rand(self.num_clusters, self.num_dimensions)
        self.pi = np.ones((self.num_clusters)) / self.num_clusters
        self.gamma = np.random.rand(self.num_samples, self.num_clusters)
        self.Sigma = np.zeros((self.num_clusters, self.num_dimensions, self.num_dimensions))
        for m in range(self.num_clusters):
            #while np.linalg.det(self.Sigma[m]) <= 0:
                #elements = np.random.rand(self.num_dimensions, self.num_dimensions)
                #matrix = np.triu(elements) + np.triu(elements, 1).T
                #matrix = np.random.rand(self.num_dimensions, self.num_dimensions)
                #self.Sigma[m] = 5 * (matrix + matrix.T) / 2.0
            self.Sigma[m] = np.eye(self.num_dimensions)
        #for m in range(self.num_clusters):
        #    while np.linalg.det(self.Sigma[m]) <= 0:
        #        self.Sigma[m] = np.random.rand(self.num_dimensions, self.num_dimensions)
        self.gamma_list = []
        self.pi_list = []
        self.mu_list = []
        self.Sigma_list = []
                
    def fit(self, x):
        for i in range(self.max_iter):
            self._e_step(x)
            print(self.Sigma)
            self._m_step(x)
            print(self.Sigma)
            print(f'number of iteration:{i + 1}\t\t\tloss:{self._log_likelihood(x)}')
        self._create_files()
    
    def _gaussian_function(self, x, mu, Sigma):
        if np.linalg.det(Sigma) == 0:
            print(Sigma)
        Lambda = np.linalg.inv(Sigma)
        #c = 1 / (np.sqrt(((2 * np.pi) ** self.num_dimensions)) * np.sqrt(np.linalg.det(Sigma)))
        c = np.sqrt(((2 * np.pi) ** self.num_dimensions)) * np.sqrt(np.linalg.det(Sigma))
        y = x - mu
        d = -y.T @ Lambda @ y / 2.0
        return np.exp(d) / c
    
    def _e_step(self, x):
        gamma = np.zeros_like(self.gamma)
        for n in range(self.num_samples):
            sum_pi = 0
            pi = np.zeros((self.num_clusters))
            for k in range(self.num_clusters):
                pi[k] = self.pi[k] * self._gaussian_function(x[n], self.mu[k], self.Sigma[k])
                sum_pi += pi[k]
            for k in range(self.num_clusters):
                gamma[n][k] = pi[k] / sum_pi
        self.gamma = gamma
        self.gamma_list.append(self.gamma)
        return
    
    def _m_step(self, x):
        pi = np.zeros_like(self.pi)
        mu = np.zeros_like(self.mu)
        Sigma = np.zeros_like(self.Sigma)
        for m in range(self.num_clusters):
            sum_gamma = 0
            sum_gamma_x = np.zeros_like(self.mu[m])
            sum_gamma_xx = np.zeros_like(self.Sigma[m])
            for n in range(self.num_samples):
                sum_gamma += self.gamma[n][m]
                sum_gamma_x += self.gamma[n][m] * x[n]
                #sum_gamma_xx += self.gamma[n][m] * (x[n].reshape((-1, 1)) @ x[n].reshape((-1, 1)).T)
                sum_gamma_xx = self.gamma[n][m] * (x[n] - self.mu[m]).reshape((-1, 1)) @ (x[n] - self.mu[m]).reshape((-1, 1)).T
            pi[m] = sum_gamma / self.num_samples
            mu[m] = sum_gamma_x / sum_gamma
            #Sigma[m] = sum_gamma_xx / sum_gamma - self.mu[m].reshape((-1, 1)) @ self.mu[m].reshape((-1, 1)).T
            Sigma[m] = sum_gamma_xx / sum_gamma
        self.pi = pi
        self.mu = mu
        self.Sigma = Sigma
        self.pi_list.append(self.pi)
        self.mu_list.append(self.mu)
        self.Sigma_list.append(self.Sigma)
        
        return
    
    def _log_likelihood(self, x):
        total_loss = 0
        for n in range(self.num_samples):
            total = 0
            for k in range(self.num_clusters):
                total += self.pi[k] * self._gaussian_function(x[n], self.mu[k], self.Sigma[k])
            total_loss += np.log(total)
        return total_loss
    
    def _create_files(self):
        df_z = pd.DataFrame(self.gamma_list)
        df_params = pd.DataFrame({'pi': self.pi_list, 'mu': self.mu_list, 'Column3': self.Sigma_list})
        df_z.to_csv("z.csv")
        df_params.to_csv("param.dat")
        return
        
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