import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from pdb import set_trace as bp

print("Pytorch version:  " + str(torch.__version__))

use_cuda = torch.cuda.is_available()

print("Use CUDA: " + str(use_cuda))

BATCH_SIZE = 64
BATCH_SIZE_TEST = 1000
EPOCHS = 10
LOG_INTERVAL = 10


torch.manual_seed(1)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE_TEST, shuffle=True, **kwargs)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        krnl_sz=3
        strd = 1
                    
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=krnl_sz, stride=strd, padding=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=krnl_sz, stride=strd, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=64, kernel_size=krnl_sz, stride=strd, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=krnl_sz, stride=strd, padding=1)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=krnl_sz, stride=strd, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=krnl_sz, stride=strd, padding=1)

        self.prelu_weight = nn.Parameter(torch.Tensor(1).fill_(0.25))

        self.fc1 = nn.Linear(3*3*512, 3)
        self.fc2 = nn.Linear(3, 2)
        self.fc3 = nn.Linear(2, 10)

    def forward(self, x):
        mp_ks=2
        mp_strd=2

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=mp_ks, stride=mp_strd)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=mp_ks, stride=mp_strd)

        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=mp_ks, stride=mp_strd)

        x = x.view(-1, 3*3*512) # Flatten

        features3d = self.fc1(x)
        features2d = self.fc2(features3d)
        x = F.prelu(features2d, self.prelu_weight)

        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1), features3d, features2d



model = Net()
model.eval()
# model.cuda()


ind = 12
image_tensor, label_tensor = test_loader.dataset[ind]
image_tensor = image_tensor.reshape(1,1,28,28)
image_tensor, label_tensor = image_tensor.to(device), label_tensor.to(device)

prediction, features3d, features2d = model(image_tensor)
prediction = np.argmax(prediction.detach().numpy())
print ("\033[92m" + "PREDICTION : " + str(prediction) + "\033[0m")

print("features3d:  " + str(features3d.detach().numpy()))
print("features2d:  " + str(features2d.detach().numpy()))



dt = test_loader.dataset.test_data.numpy()
lb = test_loader.dataset.test_labels.numpy()

print("DATA: =====")
print(dt.shape)
print("Labels: =====")
print(lb.shape)
