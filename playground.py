import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from pdb import set_trace as bp

# a = torch.Tensor([float('NaN'), 1, float('NaN'), 2, 3])
# print(a)
# a[a != a] = 0
# print(a)





x = torch.Tensor([[1., 2., 2., 3., 4., 5., 6., 7., 8., 9.], [-8.2592, -7.9788, -5.2605, -4.8818, -3.7099, -2.5116, -1.2812, -0.7652, -0.1487, -0.8805]])
indeces = torch.Tensor([6, 2])

print(x)
print("x shape: " + str(x.shape))
print(indeces)
print("indeces shape: " + str(indeces.shape))

result = torch.zeros(x.size())
for i in range(x.shape[0]):
    # print("\n")
    # print("iter: " + str(i))
    value = x[i,indeces[i].long()]
    # print(value)
    # value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
    result[i,indeces[i].long()] = value
    # print("\n")

# print("result: ")
# print(result)
