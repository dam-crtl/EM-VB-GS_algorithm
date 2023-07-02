import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

class GMM():
    def __init__(self, num_clusters):
        self.mu = None
        self.Sigma = None # Lambda^{-1}
        self.pi = None
        self.gamma = None
        self.num_clusters = num_clusters # hyperparameter
        self.num_samples = None
        self.dim = None

        return
    
    def fit(self, X, max_iter=1000, threshold=1e-02):
        num_samples = X.shape[0]
        dim = X.shape[1]
        print(dim)

        self.__init_params(dim, num_samples)

        prev = sys.float_info.max / 2
        for iter in range(max_iter):
            # E-step
            self.gamma = self.__calc_gamma(X)

            # M-step
            self.mu = self.__calc_mu(X)
            self.Sigma = self.__calc_Sigma(X)
            self.pi = self.__calc_pi(X)
            
            loss = self.__log_likelihood(X)

            print(f'iter:{iter}\tloss:{loss}')

            if np.abs(prev - loss) < threshold:
                print(f'\t>>>early stop! iter={iter}')
                break
            
            prev = loss
        
        return self.gamma

    def __init_params(self, dim, num_samples):
        self.mu = np.random.rand(self.num_clusters, dim)
        self.pi = np.random.rand(self.num_clusters)
        self.gamma = np.random.rand(num_samples, self.num_clusters)
        self.num_samples = num_samples
        self.dim = dim

        self.Sigma = np.zeros((self.num_clusters, self.dim, self.dim))
        for m in range(self.num_clusters):
            while np.linalg.det(self.Sigma[m]) <= 0:
                self.Sigma[m] = self.__gen_rand_sym_mat(self.dim)

        return
    
    def __gen_rand_sym_mat(self, size):
        elements = np.random.rand(size, size)
        matrix = np.triu(elements) + np.triu(elements, 1).T

        return matrix
    
    def __calc_gamma(self, X):
        gamma = np.zeros_like(self.gamma)
        for n in range(self.num_samples):
            p = np.zeros((self.num_clusters))
            sum = 0

            for k in range(self.num_clusters):
                p[k] = self.pi[k] * self.__gauss(X[n].T, self.mu[k].T, self.Sigma[k])
                sum += p[k]
            
            for m in range(self.num_clusters):
                gamma[n][m] = p[m] / sum
        
        return gamma
    
    def __gauss(self, x, mu, Sigma):
        Sigma_inv = np.linalg.inv(Sigma)
        Sigma_det = np.linalg.det(Sigma)
        const = 1 / np.sqrt(((2 * np.pi) ** self.dim)) * np.sqrt(Sigma_det)
        exponent = -(x - mu).T @ Sigma_inv @ (x - mu) / 2.0
        return  const * np.exp(exponent)
    
    def __calc_mu(self, X):
        mu = np.zeros_like(self.mu)

        for m in range(self.num_clusters):
            deno = np.zeros_like(mu[m])
            nume = 0
            for n in range(self.num_samples):
                deno += self.gamma[n][m] * X[n]
                nume += self.gamma[n][m]
            
            mu[m] = deno / nume
        
        return mu
    
    def __calc_Sigma(self, X):
        Sigma = np.zeros_like(self.Sigma)

        for m in range(self.num_clusters):
            deno = np.zeros_like(Sigma[m])
            nume = 0

            for n in range(self.num_samples):
                residue = (X[n] - self.mu[m]).reshape((-1, 1))

                deno += self.gamma[n][m] * residue @ residue.T
                nume += self.gamma[n][m]

            Sigma[m] = deno / nume

        return Sigma
    
    def __calc_pi(self, X):
        return np.sum(self.gamma, axis=0) / self.num_samples
    
    def __log_likelihood(self, X):
        retval = 0
        for n in range(self.num_samples):
            sum = 0
            for k in range(self.num_clusters):
                sum += self.pi[k] * self.__gauss(X[n], self.mu[k], self.Sigma[k])
            
            retval += np.log(sum)
        
        return retval
    
path_to_csv = './x.csv'
data = pd.read_csv(path_to_csv)
X = data.values
Y = [float(s) for s in data.columns]
X = np.vstack([Y, X])

gmm = GMM(num_clusters=4)
gamma = gmm.fit(X)

predicted_labels = np.argmax(gamma, axis=1)



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