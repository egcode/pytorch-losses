import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

print("Pytorch version:  " + str(torch.__version__))
use_cuda = torch.cuda.is_available()
print("Use CUDA: " + str(use_cuda))

# Cosface
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd.function import Function
import math

from pdb import set_trace as bp

BATCH_SIZE = 100
FEATURES_DIM = 3
NUM_CLASSES = 10

BATCH_SIZE_TEST = 1000
EPOCHS = 20
LOG_INTERVAL = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        krnl_sz=3
        strd = 1
                    
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=krnl_sz, stride=strd, padding=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=krnl_sz, stride=strd, padding=1)
        self.prelu1_1 = nn.PReLU()
        self.prelu1_2 = nn.PReLU()
        
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=64, kernel_size=krnl_sz, stride=strd, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=krnl_sz, stride=strd, padding=1)
        self.prelu2_1 = nn.PReLU()
        self.prelu2_2 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=krnl_sz, stride=strd, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=krnl_sz, stride=strd, padding=1)
        self.prelu3_1 = nn.PReLU()
        self.prelu3_2 = nn.PReLU()

        self.prelu_weight = nn.Parameter(torch.Tensor(1).fill_(0.25))

        self.fc1 = nn.Linear(3*3*512, 3)
        # self.fc2 = nn.Linear(3, 2)
        self.fc3 = nn.Linear(3, 10)

    def forward(self, x):
        mp_ks=2
        mp_strd=2

        x = self.prelu1_1(self.conv1(x))
        x = self.prelu1_2(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=mp_ks, stride=mp_strd)

        x = self.prelu2_1(self.conv3(x))
        x = self.prelu2_2(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=mp_ks, stride=mp_strd)

        x = self.prelu3_1(self.conv5(x))
        x = self.prelu3_2(self.conv6(x))
        x = F.max_pool2d(x, kernel_size=mp_ks, stride=mp_strd)

        x = x.view(-1, 3*3*512) # Flatten
        features3d = F.prelu(self.fc1(x), self.prelu_weight)
        x = self.fc3(features3d)
    
        return features3d, x



class Arcface_loss(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    # def __init__(self, num_classes=NUM_CLASSES, feat_dim=FEATURES_DIM, s=7.00, m=0.2):
    def __init__(self, feat_dim, num_classes, device, s=7., m=0.50, easy_margin=False):
        super(Arcface_loss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.device = device

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.clamp(-1,1) # for numerical stability

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size()).to(device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
        
    
# class Arcface_loss(nn.Module):
    # def __init__(self, num_classes=NUM_CLASSES, feat_dim=FEATURES_DIM, s=7.00, m=0.2):
    
#         super(Arcface_loss, self).__init__()
#         self.num_classes = num_classes
#         self.kernel = nn.Parameter(torch.Tensor(feat_dim, num_classes))
#         # initial kernel
#         self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
#         self.m = m # the margin value, default is 0.5
#         self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.mm = self.sin_m * m  # issue 1
#         self.threshold = math.cos(math.pi - m)
#     def forward(self, embbedings, label):
#         # weights norm
#         nB = len(embbedings)
#         kernel_norm = l2_norm(self.kernel,axis=0)
#         # cos(theta+m)
#         cos_theta = torch.mm(embbedings,kernel_norm)
# #         output = torch.mm(embbedings,kernel_norm)
#         cos_theta = cos_theta.clamp(-1,1) # for numerical stability
#         cos_theta_2 = torch.pow(cos_theta, 2)
#         sin_theta_2 = 1 - cos_theta_2
#         sin_theta = torch.sqrt(sin_theta_2)
#         cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
#         # this condition controls the theta+m should in range [0, pi]
#         #      0<=theta+m<=pi
#         #     -m<=theta<=pi-m
#         cond_v = cos_theta - self.threshold
#         cond_mask = cond_v <= 0
#         keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
#         cos_theta_m[cond_mask] = keep_val[cond_mask]
#         output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
#         idx_ = torch.arange(0, nB, dtype=torch.long)
#         output[idx_, label] = cos_theta_m[idx_, label]
#         output *= self.s # scale up in order to make softmax work, first introduced in normface
#         return output
# def l2_norm(input,axis=1):
#     norm = torch.norm(input,2,axis,True)
#     output = torch.div(input, norm)
#     return output

  

def train(model, device, train_loader, loss_softmax, loss_arcface, optimizer_nn, optimzer_arcface, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # optimizer.zero_grad()
        # output,_,_ = model(data)
        
        # loss = loss_function(output, target)
        
        # loss.backward()
        # optimizer.step()

        features, _ = model(data)
        logits = loss_arcface(features, target)
        loss = loss_softmax(logits, target)

        _, predicted = torch.max(logits.data, 1)
        accuracy = (target.data == predicted).float().mean()

        optimizer_nn.zero_grad()
        optimzer_arcface.zero_grad()

        loss.backward()

        optimizer_nn.step()
        optimzer_arcface.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, loss_softmax, loss_arcface):
    model.eval()
    # test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            feats, _ = model(data)
            logits = loss_arcface(feats, target)
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target.data).sum()

    # print('Test Accuracy of the model on the 10000 test images: %f %%' % (100 * correct / total))


    print('\nTest set:, Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))    

    #         output,_,_ = model(data)
            
    #         test_loss += loss_function(output, target).item() # sum up batch loss

    #         pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))    

###################################################################

torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")

####### Data setup

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

####### Model setup

model = Net().to(device)
loss_softmax = nn.CrossEntropyLoss().to(device)
loss_arcface = Arcface_loss(num_classes=NUM_CLASSES, feat_dim=FEATURES_DIM, device=device).to(device)
       
# optimzer nn
optimizer_nn = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
sheduler_nn = lr_scheduler.StepLR(optimizer_nn, 20, gamma=0.5)

# optimzer cosface or arcface
optimzer_arcface = optim.SGD(loss_arcface.parameters(), lr=0.01)
sheduler_arcface = lr_scheduler.StepLR(optimzer_arcface, 20, gamma=0.5)


for epoch in range(1, EPOCHS + 1):
    sheduler_nn.step() 
    sheduler_arcface.step()

    train(model, device, train_loader, loss_softmax, loss_arcface, optimizer_nn, optimzer_arcface, epoch)
    test(model, device, test_loader, loss_softmax, loss_arcface)

torch.save(model.state_dict(),"mnist_cnn-arcface.pt")        
torch.save(loss_arcface.state_dict(),"mnist_loss-arcface.pt")        
