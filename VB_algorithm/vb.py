import csv
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from scipy.special import digamma, gamma, logsumexp
from mpl_toolkits.mplot3d import Axes3D
import sys

txt_dir = "./x.csv"

with open(txt_dir) as f:
  reader = csv.reader(f)
  l = [row for row in reader]
  for i in range(len(l)):
    for j in range(len(l[i])):
      l[i][j] = float(l[i][j])
l = np.array(l)
class VB_algorithm_GMM():
  def __init__(self, n_clusters=4, alpha0=0.1):
    self.n_clusters = n_clusters
    self.alpha0 = alpha0
    self.beta0 = 1.0
    self.threshold = 1e-2
    
  def _init_params(self, X):
    np.random.seed(seed=42)
    self.n_sample, self.n_feature = X.shape
    self.m0 = np.random.randn(self.n_feature)
    self.W0 = np.eye(self.n_feature)
    self.v0 = np.array([self.n_feature])
    self.gamma = np.ones(self.n_clusters) * (self.n_sample / self.n_clusters)
    self.alpha = np.ones(self.n_clusters) * self.alpha0
    self.beta = np.ones(self.n_clusters) * self.beta0
    self.m = np.random.randn(self.n_clusters, self.n_feature)
    self.W = np.tile(self.W0, (self.n_clusters, 1, 1)) 
    self.v = np.ones(self.n_clusters) * self.n_feature
    self.Sigma = np.zeros((self.n_clusters, self.n_feature, self.n_feature))
    for k in range(self.n_clusters):
      self.Sigma[k] = np.linalg.inv(self.v[k] * self.W[k])
    self.mu = np.random.randn(self.n_clusters, self.n_feature)
  
  def expectation(self, X):
    pi = digamma(self.alpha) - digamma(self.alpha.sum())
    Lambda_tilde = np.zeros((self.n_clusters))
    for k in range(self.n_clusters):
      digamma_sum = 0
      for i in range(self.n_feature):
        digamma_sum += digamma((self.v[k] + 1 - i)/2)
      Lambda_tilde[k] = digamma_sum + self.n_feature * np.log(2) + np.log(np.linalg.det(self.W[k]))
    rho = np.zeros((self.n_sample, self.n_clusters))
    for n in range(self.n_sample):
      for k in range(self.n_clusters):         
        residual = X[n] - self.m[k]
        rho[n][k] = pi[k] + Lambda_tilde[k]/2.0 - (self.n_feature/(2*self.beta[k])) - (self.v[k]/2.0)*(residual.T @ self.W[k] @ residual)
    r_log = rho - logsumexp(rho, axis=1).reshape(-1, 1)
    r = np.exp(r_log)
    #r[np.isnan(r)] = 1.0 / (self.n_clusters)
    return r
  
  def maximization(self, X, gamma):
      self.gamma = np.sum(gamma, axis=0, keepdims=True).T
      gamma_x = (gamma.T @ X) / self.gamma
      gamma_xx = np.zeros((self.n_sample, self.n_clusters, self.n_feature, self.n_feature))
      for n in range(self.n_sample):
        for k in range(self.n_clusters):
          residual = (X[n] - gamma_x[k]).reshape(-1, 1)
          gamma_xx[n][k] = gamma[n][k] * residual @ residual.T
      S = np.sum(gamma_xx, axis=0) / self.gamma[:, None]
      self.alpha = self.alpha0 + self.gamma
      self.beta = self.beta0 + self.gamma
      for k in range(self.n_clusters):  
        self.m[k] = (1/self.beta[k]) * (self.beta0 * self.m0 + self.gamma[k] * gamma_x[k])
      for k in range(self.n_clusters):
        residual = (gamma_x[k] - self.m0).reshape(-1, 1)
        self.W[k] = np.linalg.inv(np.linalg.inv(self.W0) + self.gamma[k] * S[k] + ((self.beta0*self.gamma[k]) / (self.beta0 + self.gamma[k])) * residual@residual.T)
        self.v[k] = self.v0 + self.gamma[k]
      pi = self.alpha / np.sum(self.alpha)
      return pi
  
  def multivariate_gaussian(self, x, mu, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_det = np.linalg.det(Sigma)
    exponent = -(x - mu).T @ Sigma_inv @ (x - mu) / 2.0
    const = 1 / ((np.sqrt(2*np.pi)**self.n_feature) * np.sqrt(Sigma_det))
    return const * np.exp(exponent)
  
  def mixed_gaussian(self, X, mu, Sigma, pi):
    output = np.zeros((self.n_sample))
    for i in range(self.n_clusters):
      for j in range(self.n_sample):
        output[j] += pi[i] * self.multivariate_gaussian(X[j], mu[i], Sigma[i])
    return output
  
  def log_likelihood(self, X, pi):
    for i in range(self.n_clusters):
      self.Sigma[i] = np.linalg.inv(self.v[i] * self.W[i])
    self.mu = self.m
    gaussians = self.mixed_gaussian(X, self.mu, self.Sigma, pi)
    logs = 0
    for i in range(self.n_sample):
      logs += np.log(gaussians[i])
    return logs
  
  def fit(self, X, iter_max, thr):
    self._init_params(X)
    pi = np.array([1/self.n_clusters for i in range(self.n_clusters)])
    prelikelihood = -1000000
    for i in range(iter_max):
      r = self.expectation(X)
      pi = self.maximization(X, r)
      likelihood = self.log_likelihood(X, pi)
      print(f"number of iteration {i + 1} : log-likelihood {likelihood}")
      if np.abs(likelihood - prelikelihood) < self.threshold:
        print("early stop!")
        return r, pi, self.mu, self.Sigma
      else:
        prelikelihood = likelihood
        
  def classify(self, X):
    return np.argmax(self.expectation(X), 1)   

model = VB_algorithm_GMM(n_clusters=4, alpha0=0.01)
r, pi, Mu, Sigma = model.fit(l, iter_max=100, thr = 0.01)
labels = model.classify(l)


cm = plt.get_cmap("tab10")
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
N = l.shape[0]

for n in range(N):
  ax.plot([l[n][0]], [l[n][1]], [l[n][2]],  "o", color=cm(labels[n]))
ax.view_init(elev=30, azim=45)
plt.show()