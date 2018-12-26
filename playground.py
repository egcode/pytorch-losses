import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


a = torch.Tensor([float('NaN'), 1, float('NaN'), 2, 3])
print(a)
a[a != a] = 0
print(a)


