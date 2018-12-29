import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
print("Pytorch version:  " + str(torch.__version__))
use_cuda = torch.cuda.is_available()
print("Use CUDA: " + str(use_cuda))
from pdb import set_trace as bp
torch.set_printoptions(threshold=1000000)

BATCH_SIZE = 64
BATCH_SIZE_TEST = 1000
EPOCHS = 20
LOG_INTERVAL = 10
NUM_OF_CLASSES = 10


torch.manual_seed(1)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
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
        
        return x, features3d, features2d


class CrossEntropyCustom(nn.Module):

    def __init__(self):
        super(CrossEntropyCustom, self).__init__()
        self.sm = nn.Softmax()
        self.lsm = nn.LogSoftmax()

        self.nll = nn.NLLLoss()

    def forward(self, input, target):

        ######################################################
        #################### Log Softmax #########################

        # Stable Logsoftmax - 
        b = torch.max(input)
        presum = torch.exp(input - b)
        prelog = presum.sum(-1).unsqueeze(-1)
        prelog = prelog.clamp(min=1e-33, max=1e+33) # for numerical stability
        log = torch.log(prelog)
        log_probabilities = (input - b) - log

        # --- Pytorch WORKING Logsoftmax
        # log_probabilities = self.lsm(input).cpu()


        ######################################################
        #################### NLLLoss #########################

        target = target.cpu() ## To remove error on gpu
        log_probabilities = log_probabilities.cpu() ## To remove error on gpu

        m = target.shape[0]

        ## NLLLoss V1
        cross_entropy = torch.zeros(log_probabilities.size())
        for i in range(m):
            value = log_probabilities[i,target[i].long()]
            cross_entropy[i,target[i].long()] = value

        ## NLLLoss V2
        # target_one_hot = torch.zeros(len(target), NUM_OF_CLASSES).scatter_(1, target.unsqueeze(1), 1.)        
        # cross_entropy = torch.addcmul(torch.zeros(log_probabilities.size()), 1., log_probabilities, target_one_hot)

        loss = -(1./m) * torch.sum(cross_entropy)
        return loss


def train(model, loss_custom, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output,_,_ = model(data)
        
        loss = loss_custom(output, target)

        optimizer.zero_grad() # clear previous gradients
        loss.backward() # compute gradients of all variables wrt loss

        df = nn.CrossEntropyLoss()
        optimizer.step() # perform updates using calculated gradients

        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, loss_custom, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,_,_ = model(data)
            
            # test_loss += loss_function(output, target).item() # sum up batch loss
            test_loss += loss_custom(output, target).item() # sum up batch loss

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



model = Net()
# model.load_state_dict(torch.load("mnist_cnn-softmax2.pt"))
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

loss_custom = CrossEntropyCustom().to(device)

for epoch in range(1, EPOCHS + 1):
    train(model, loss_custom, device, train_loader, optimizer, epoch)
    test(model, loss_custom, device, test_loader)
torch.save(model.state_dict(),"mnist_cnn-softmax2.pt")        
