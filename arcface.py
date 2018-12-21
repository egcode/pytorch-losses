import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import math
import numpy as np
from pdb import set_trace as bp

print("Pytorch version:  " + str(torch.__version__))
use_cuda = torch.cuda.is_available()
print("Use CUDA: " + str(use_cuda))

FEATURE_SIZE = 3
BATCH_SIZE = 64
BATCH_SIZE_TEST = 1000
EPOCHS = 50
LOG_INTERVAL = 10
NUM_OF_CLASSES = 10
LR = 1e-1  # initial learning rate
LR_STEP = 10
LR_DECAY = 0.95  # when val_loss increase, LR = LR*LR_DECAY

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
        self.fc1 = nn.Linear(3*3*512, FEATURE_SIZE)
        self.fc3 = nn.Linear(FEATURE_SIZE, 10)
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
        x = F.prelu(features3d, self.prelu_weight)
        x = self.fc3(x)
        return x, features3d

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, device, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.device = device

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

def train(model, metric_fc, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        pred, features3d = model(data)

        output = metric_fc(features3d, labels)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, metric_fc, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            pred,features3d = model(data)
            
            output = metric_fc(features3d, labels)
            test_loss += criterion(output, labels)

            # test_loss += centerLoss(output, target, device, features3d)

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
  
    print('\nTest set: Average loss: {}, Accuracy: {}/{} ({}%)\n'.format(
        str(test_loss), str(correct), str(len(test_loader.dataset)),
        str(100. * correct / len(test_loader.dataset))))

##########################################################

model = Net().to(device)
model.to(device)

metric_fc = ArcMarginProduct(FEATURE_SIZE, NUM_OF_CLASSES, device, s=30, m=0.5, easy_margin=False).to(device)
metric_fc.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=LR, weight_decay=LR_DECAY)

scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=0.1)

for epoch in range(1, EPOCHS + 1):
    train(model, metric_fc, criterion, device, train_loader, optimizer, epoch)
    test(model, metric_fc, criterion, device, test_loader)

torch.save(model.state_dict(),"mnist_cnn-arcface-loss.pt")
        