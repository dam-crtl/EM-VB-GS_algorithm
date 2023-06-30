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
  def __init__(self, X, n_clusters=4, alpha0=0.1):
    self.n_clusters = n_clusters
    self.alpha0 = alpha0
    self.n_sample, self.n_feature = X.shape
    self.m0 = np.random.randn(self.n_feature)
    self.beta0 = np.array([1.0])
    self.W0 = np.eye(self.n_feature)
    self.n_sampleu0 = np.array([self.n_feature])
    self.n_sample_k = (self.n_sample / self.n_clusters) + np.zeros(self.n_clusters)
    self.alpha = np.ones(self.n_clusters) * self.alpha0
    self.beta = np.ones(self.n_clusters) * self.beta0
    self.m = np.random.randn(self.n_clusters, self.n_feature)
    self.W = np.tile(self.W0, (self.n_clusters, 1, 1))
    self.n_sampleu = np.ones(self.n_clusters)*self.n_feature
    self.Sigma = np.zeros((self.n_clusters, self.n_feature, self.n_feature))
    for k in range(self.n_clusters):
      self.Sigma[k] = np.linalg.inv(self.n_sampleu[k] * self.W[k])
    self.mu = self.m
  
  def expectation(self, X):
    pi = digamma(self.alpha) - digamma(self.alpha.sum())
    Lambda_tilde = np.zeros((self.n_clusters))
    for k in range(self.n_clusters):
      digamma_sum = np.array([])
      for i in range(self.n_feature):
        digamma_sum.appned(digamma((self.n_sampleu[k] + 1 - i)/2))
      Lambda_tilde[k] = np.sum(digamma_sum) + self.n_feature * np.log(2) + np.log(la.det(self.W[k]))
    rho = np.zeros((self.n_sample, self.n_clusters))
    for n in range(self.n_sample):
      for k in range(self.n_clusters):         
        gap = (X[n] - self.m[k]).reshape(-1, 1)
        rho[n][k] = pi[k] + Lambda_tilde[k]/2.0 - (self.n_feature/(2*self.beta[k])) -(self.n_sampleu[k]/2)*(gap.T@self.W[k]@gap)
    r_log = rho - logsumexp(rho, axis=1)[:,None]
    r = np.exp(r_log)
    r[np.isnan(r)] = 1.0 / (self.n_clusters)
    return r
  
  def maximization(self, X, r):
      self.n_sample_k = np.sum(r, axis=0, keepdims=True).T
      gamma_x = (r.T @ X) / self.n_sample_k
      S_list = np.zeros((self.n_sample, self.n_clusters, self.n_feature, self.n_feature))
      for n in range(self.n_sample):
        for k in range(self.n_clusters):
          gap = (X[n] - gamma_x[k])[:, None]
          S_list[n][k] = r[n][k] * gap @ gap.T
      S = np.sum(S_list, axis=0) / self.n_sample_k[:,None]
      self.alpha = self.alpha0 + self.n_sample_k
      self.beta = self.beta0 + self.n_sample_k
      for k in range(self.n_clusters):  
        self.m[k] = (1/self.beta[k]) * (self.beta0 * self.m0 + self.n_sample_k[k] * gamma_x[k])
      for k in range(self.n_clusters):
        gap = (gamma_x[k] - self.m0)[:, None]
        A = np.linalg.inv(self.W0)
        B = self.n_sample_k[k] * S[k]
        C = ((self.beta0*self.n_sample_k[k]) / (self.beta0 + self.n_sample_k[k])) * gap@gap.T
        self.W[k] = np.linalg.inv(A + B + C)
        self.n_sampleu[k] = self.n_sampleu0 + self.n_sample_k[k]
      pi = self.alpha / np.sum(self.alpha, keepdims=True)
      return pi
  
  def gaussian_function(self, x, mu, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_det = np.linalg.det(Sigma)
    exponent = -(x - mu).T @ Sigma_inv.T @ (x - mu) / 2.0
    const = 1 / ((np.sqrt(2*np.pi)**self.n_feature) * np.sqrt(Sigma_det))
    return const * np.exp(exponent)
  
  def mixed_gaussian(self, X, mu, Sigma, pi):
    output = np.array([])
    eps = np.spacing(1)
    Eps = eps*np.eye(sigma.shape[0])
    sigma_inv = np.linalg.inv(sigma)
    sigma_det = np.linalga.det(sigma)
    for i in range(self.n_sample):
      output = append(, self.calc(X[i], mu, sigma_inv, sigma_det))
    return output
  
  def mix_gauss(self, X, Mu, Sigma, Pi):
    output = np.array([Pi[i]*self.gauss(X, Mu[i], Sigma[i]) for i in range(self.n_clusters)])
    return output, np.sum(output, 0)[None,:]
  
  def log_likelihood(self, X, pi):
    for i in range(self.n_clusters):
      self.Sigma[i] = la.inv(self.n_sampleu[i] * self.W[i])
    self.mu = self.m
    _, out_sum = self.mix_gauss(X, self.mu, self.Sigma, pi)
    logs = np.array([np.log(out_sum[0][n]) for n in range(self.n_sample)])
    return np.sum(logs)
  
  def fit(self, X, iter_max, thr):
    self.init_params(X)
    likelihood_list = np.array([])
    pi = np.array([1/self.n_clusters for i in range(self.n_clusters)])
    likelihood_list = np.append(likelihood_list, self.log_likelihood(X, pi))
    count = 0
    for i in range(iter_max):
      r = self.expectation(X)
      pi = self.maximization(X, r)
      likelihood = self.log_likelihood(X, pi)
      print(f"iteration {i + 1} : log_likelihood: {likelihood}")
      if np.abs(likelihood_list[count] - likelihood_list[count+1]) < thr:
        return count+1, likelihood_list, r, pi, self.mu, self.Sigma
      else:
        print("Previous log-likelihood gap:" + str(np.abs(likelihood_list[count] - likelihood_list[count+1])))
        count += 1
        
  def classify(self, X):
    return np.argmax(self.expectation(X), 1)   

model = VBGMM(n_clusters=4, alpha0=0.01)
n_iter, likelihood_list, r, pi, Mu, Sigma = model.fit(l, iter_max=100, thr = 0.01)
labels = model.classify(l)
print(n_iter)
print(likelihood_list)


cm = plt.get_cmap("tab10")
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
N = l.shape[0]

for n in range(N):
  ax.plot([l[n][0]], [l[n][1]], [l[n][2]],  "o", color=cm(labels[n]))
ax.view_init(elev=30, azim=45)
plt.show()