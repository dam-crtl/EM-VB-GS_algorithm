# EM-VB_algorithm

This "submission" file contains "EM_algorithm" and "VB_algorithm".
Here, I will explain about each folder.
## environment 
OS : Mac OS 
used modules:
numpy 1.25.0
pandas 1.5.2
scipy 1.11.0
matplotlib 3.6.2

## contents
### EM_algorithm
This is the file for implementaion of EM algorithm.
This file contains "x.csv", "z.csv", "your_executable_command", "EM_algorithm.py"
##### x.csv:
This file contains input dataset.
##### z.csv:
This file contains posterior probabilities of latent variables.This file does not have both index and header
##### params.dat
This file contains the value of parameters \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}
##### executable_your_command:
This file contains the information of command that are used to execute "EM_algorithm.py". 
##### EM_algorithm.py:
This file contains implementation of EM algorithm. By executing this file, you can get z.csv and params.dat. Furthermore, you can see how the points of x.csv is classified when using EM algorithm.

### VB_algorithm
This is the file for implementaion of VB algorithm.
This file contains "x.csv", "z.csv", "your_executable_command", "VB_algorithm.py"
##### x.csv:
This file contains input dataset.
##### z.csv:
This file contains posterior probabilities of latent variables.This file does not have both index and header
##### params.dat
This file contains the value of parameters \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}
##### executable_your_command:
This file contains the information of command that are used to execute "VB_algorithm.py". 
##### VB_algorithm.py:
This file contains implementation of VB algorithm. By executing this file, you can get z.csv and params.dat. Furthermore, you can see how the points of x.csv is classified when using VB algorithm.