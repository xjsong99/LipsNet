# LipsNet
This is a PyTorch implementation of LipsNet.

The paper is accepted at ICML 2023 with the title '*LipsNet: A Smooth and Robust Neural Network with Adaptive Lipschitz Constant for High Accuracy Optimal Control*'.

The overall structure is shown below:
![](figures/structure.png)

## Requirements
The version of PyTorch should be higher than 1.11 and lower than 2.3,
as we incorporate *functorch.jacrev* and *functorch.vmap* methods.

## How to use
We package LipsNet as a PyTorch module.

Practitioners can easily use it just like using MLP.

```
from lipsnet import LipsNet

# declare
net = LipsNet(...)

# training
net.train()
out = net(input)
...
loss.backward()
net.eval()

# evaluation
net.eval()
out = net(input)
```

More details can be found in *lipsnet.py*.