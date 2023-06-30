import csv
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from scipy.special import digamma, gamma, logsumexp
import matplotlib.pyplot as plt
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

class VBGMM():
  def __init__(self, K=6, alpha0=0.1):
    self.K = K
    self.alpha0 = alpha0
  
  def init_params(self, X):
    self.N, self.D = X.shape
    self.m0 = np.random.randn(self.D)
    self.beta0 = np.array([1.0])
    self.W0 = np.eye(self.D)
    self.nu0 = np.array([self.D])
    
    self.N_k = (self.N / self.K) + np.zeros(self.K)
    
    self.alpha = np.ones(self.K) * self.alpha0
    self.beta = np.ones(self.K) * self.beta0
    self.m = np.random.randn(self.K, self.D)
    self.W = np.tile(self.W0, (self.K, 1, 1))
    self.nu = np.ones(self.K)*self.D
    
    self.Sigma = np.zeros((self.K, self.D, self.D))
    for k in range(self.K):
      self.Sigma[k] = la.inv(self.nu[k] * self.W[k])
    self.Mu = self.m
  
  def e_step(self, X):
    pi = digamma(self.alpha) - digamma(self.alpha.sum())
    Lambda_tilde = np.zeros((self.K))
    for k in range(self.K):
      digamma_sum = np.array([])
      for i in range(self.D):
        digamma_sum = np.append(digamma_sum, digamma((self.nu[k] + 1 - i)/2))
      A = np.sum(digamma_sum)
      B = self.D * np.log(2)
      C = np.log(la.det(self.W[k]))
      Lambda_tilde[k] = A + B + C
    rho = np.zeros((self.N, self.K))
    for n in range(self.N):
      for k in range(self.K):         
        gap = (X[n] - self.m[k])[:, None]
        A = -(self.D/(2*self.beta[k]))
        B = -(self.nu[k]/2)*(gap.T@self.W[k]@gap)
        rho[n][k] = pi[k] + 0.5*Lambda_tilde[k] + A + B
    r_log = rho - logsumexp(rho, axis=1)[:,None]
    r = np.exp(r_log)
    r[np.isnan(r)] = 1.0 / (self.K)
    return r
  
  def m_step(self, X, r):
      self.N_k = np.sum(r, axis=0, keepdims=True).T
      barx = (r.T @ X) / self.N_k
      S_list = np.zeros((self.N, self.K, self.D, self.D))
      for n in range(self.N):
        for k in range(self.K):
          gap = (X[n] - barx[k])[:, None]
          S_list[n][k] = r[n][k] * gap @ gap.T
      S = np.sum(S_list, axis=0) / self.N_k[:,None]
      self.alpha = self.alpha0 + self.N_k
      self.beta = self.beta0 + self.N_k
      for k in range(self.K):  
        self.m[k] = (1/self.beta[k]) * (self.beta0 * self.m0 + self.N_k[k] * barx[k])
      for k in range(self.K):
        gap = (barx[k] - self.m0)[:, None]
        A = la.inv(self.W0)
        B = self.N_k[k] * S[k]
        C = ((self.beta0*self.N_k[k]) / (self.beta0 + self.N_k[k])) * gap@gap.T
        self.W[k] = la.inv(A + B + C)
        self.nu[k] = self.nu0 + self.N_k[k]
      pi = self.alpha / np.sum(self.alpha, keepdims=True)
      return pi
  
  def calc(self, x, Mu, sigma_inv, sigma_det):
    exp = -0.5*(x - Mu).T@sigma_inv.T@(x - Mu)
    denomin = np.sqrt(sigma_det)*(np.sqrt(2*np.pi)**self.D)
    return np.exp(exp)/denomin
  
  def gauss(self, X, Mu, sigma):
    output = np.array([])
    eps = np.spacing(1)
    Eps = eps*np.eye(sigma.shape[0])
    sigma_inv = la.inv(sigma)
    sigma_det = la.det(sigma)
    for i in range(self.N):
      output = np.append(output, self.calc(X[i], Mu, sigma_inv, sigma_det))
    return output
  
  def mix_gauss(self, X, Mu, Sigma, Pi):
    output = np.array([Pi[i]*self.gauss(X, Mu[i], Sigma[i]) for i in range(self.K)])
    return output, np.sum(output, 0)[None,:]
  
  def log_likelihood(self, X, pi):
    for i in range(self.K):
      self.Sigma[i] = la.inv(self.nu[i] * self.W[i])
    self.Mu = self.m
    _, out_sum = self.mix_gauss(X, self.Mu, self.Sigma, pi)
    logs = np.array([np.log(out_sum[0][n]) for n in range(self.N)])
    return np.sum(logs)
  
  def fit(self, X, iter_max, thr):
    self.init_params(X)
    log_list = np.array([])
    pi = np.array([1/self.K for i in range(self.K)])
    log_list = np.append(log_list, self.log_likelihood(X, pi))
    count = 0
    for i in range(iter_max):
      r = self.e_step(X)
      pi = self.m_step(X, r)
      log_list = np.append(log_list, self.log_likelihood(X, pi))
      if np.abs(log_list[count] - log_list[count+1]) < thr:
        return count+1, log_list, r, pi, self.Mu, self.Sigma
      else:
        print("Previous log-likelihood gap:" + str(np.abs(log_list[count] - log_list[count+1])))
        count += 1
        
  def classify(self, X):
    return np.argmax(self.e_step(X), 1)   

model = VBGMM(K=4, alpha0=0.01)
n_iter, log_list, r, pi, Mu, Sigma = model.fit(l, iter_max=100, thr = 0.01)
labels = model.classify(l)
print(n_iter)
print(log_list)


cm = plt.get_cmap("tab10")
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
N = l.shape[0]

for n in range(N):
  ax.plot([l[n][0]], [l[n][1]], [l[n][2]],  "o", color=cm(labels[n]))
ax.view_init(elev=30, azim=45)
plt.show()