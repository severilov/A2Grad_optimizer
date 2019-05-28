# A2Grad

This is the code associated with Adaptive and Accelerated SGD algorithm used in the paper [Optimal Adaptive and Accelerated Stochastic Gradient Descent](https://arxiv.org/abs/1810.00553), oct. 2018

## Usage:
In a manner similar to using any usual optimizer from the pytorch toolkit, it is also possible to use the A2Grad optimizer with little effort.
First, we require importing the optimizer through the following command:
```
from optimizers import *
```
Next, an A2Grad optimizer working with a given pytorch `model` can be invoked using the following command (depending on which realisation you want):
```
optimizer = A2GradUni(model.parameters(), beta=10, lips=10)
optimizer = A2GradInc(model.parameters(), beta=10, lips=10)
optimizer = A2GradExp(model.parameters(), beta=1, lips=10, rho=0.9)
```

## Our experiments:
We implemented 3 realisations of A2Grad from the paper and compared it with Adam, AMSGrad, accelerated SGD (variant from this [paper](https://arxiv.org/abs/1803.05591)) and adaptive SGD (Spokoiny's practical variant)
* optimizers.py contains all implementations of tested optimizers, including 3 different variants of A2Grad (A2GradUni, A2GradInc, A2GradExp)
* MNIST.ipynb contains all experiments on the MNIST dataset, tested models: logistic regression and two-layer neural network
* CIFAR10.ipynb contains all experiments on the CIFAR10 dataset tested models: Cifarnet (~~and Vgg16~~)
* plot_results.ipynb contains all visualized results from MNIST and CIFAR10 experiments
