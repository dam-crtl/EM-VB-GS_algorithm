# EM-VB_algorithm

This "submission" file contains "EM_algorithm" and "VB_algorithm".
Here, I will explain about each folder.
## environment 
#### used modules:
numpy 1.25.0\
pandas 1.5.2\
scipy 1.11.0\
matplotlib 3.6.2

## contents of "EM_algorithm" folder and "VB_algorithm" folder
### EM_algorithm
This is the file for implementaion of EM algorithm.
This file contains "x.csv", "z.csv", "EM_algorithm.py"
#### x.csv:
This file contains input dataset.
#### z.csv:
This file contains posterior probabilities of latent variables.This file does not have both index and header
#### params.dat
This file contains the value of parameters $\boldsymbol{\pi}$, $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$
#### EM_algorithm.py:
This file contains implementation of EM algorithm. By executing this file, you can get "z.csv" and "params.dat". Furthermore, you can see how the points of "x.csv" is classified when using EM algorithm.
When you execute this file, please enter the following command line.

```console
$ python EM_algorithm.py x.csv z.csv params.dat
``` 

If you want to get the figure when the number of the cluster set to three, you have to execute this file after commenting out the lines from 102 to 126 and uncommentng the lines from 130 to 144.
If you want to get the figure when the number of the cluster set to three, you have to execute this file after commenting out the lines from 102 to 126 and uncommentng the lines from 148 to 162.\
The lines from 102 to 126 are the below.
```python
    #assuming the number of cluster is 4
    n_clusters = 4
    model = EM_algorithm_GMM(n_clusters=n_clusters)
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
``` 
The lines from 130 to 144 are the below.
```python
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
``` 
The lines from 148 to 162 are the below.
```python
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
``` 
### VB_algorithm
This is the file for implementaion of VB algorithm.
This file contains "x.csv", "z.csv", "your_executable_command", "VB_algorithm.py"
#### x.csv:
This file contains input dataset.
#### z.csv:
This file contains posterior probabilities of latent variables.This file does not have both index and header
#### params.dat
This file contains the value of parameters $\boldsymbol{\pi}$, $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$
#### VB_algorithm.py:
This file contains implementation of VB algorithm. By executing this file, you can get "z.csv" and "params.dat". Furthermore, you can see how the points of "x.csv" is classified when using VB algorithm.
When you execute this file, please enter the following command line.

```console
$ python VB_algorithm.py x.csv z.csv params.dat
``` 
If you want to get only the figure when the number of the cluster set to three, you have to execute this file after commenting out the lines from 137 to 161 and uncommentng the lines from 165 to 179.
If you want to get only the figure when the number of the cluster set to three, you have to execute this file after commenting out the lines from 137 to 161 and uncommentng the lines from 183 to 197.\
The lines from 137 to 161 are the below.
```python
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
``` 
The lines from 165 to 179 are the below.
```python
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
``` 
The lines from 183 to 197 are the below.
```python
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
``` 

### GS_algorithm
This is the file for implementaion of GS algorithm.
This file contains "x.csv", "z.csv", "your_executable_command", "GS_algorithm.py"
#### x.csv:
This file contains input dataset.
#### z.csv:
This file contains posterior probabilities of latent variables.This file does not have both index and header
#### params.dat
This file contains the value of parameters $\boldsymbol{\pi}$, $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$
#### GS_algorithm.py:
This file contains implementation of VB algorithm. By executing this file, you can get "z.csv" and "params.dat". Furthermore, you can see how the points of "x.csv" is classified when using VB algorithm.
When you execute this file, please enter the following command line.

```console
$ python GS_algorithm.py x.csv z.csv params.dat
``` 
If you want to get only the figure when the number of the cluster set to three, you have to execute this file after commenting out the lines from 133 to 157 and uncommentng the lines from 161 to 175.
If you want to get only the figure when the number of the cluster set to three, you have to execute this file after commenting out the lines from 133 to 157 and uncommentng the lines from 179 to 193.\
The lines from 133 to 157 are the below.
```python
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
``` 
The lines from 161 to 175 are the below.
```python
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
``` 
The lines from 179 to 193 are the below.
```python
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
``` 