import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from pdb import set_trace as bp

from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np

BATCH_SIZE = 100
FEATURES_DIM = 3

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
#         self.prelu1_1 = nn.PReLU()
#         self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
#         self.prelu1_2 = nn.PReLU()
#         self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
#         self.prelu2_1 = nn.PReLU()
#         self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
#         self.prelu2_2 = nn.PReLU()
#         self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
#         self.prelu3_1 = nn.PReLU()
#         self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
#         self.prelu3_2 = nn.PReLU()
#         self.preluip1 = nn.PReLU()
#         self.ip1 = nn.Linear(128 * 3 * 3, FEATURES_DIM)
#         self.ip2 = nn.Linear(FEATURES_DIM, 10)

#     def forward(self, x):
#         x = self.prelu1_1(self.conv1_1(x))
#         x = self.prelu1_2(self.conv1_2(x))
#         x = F.max_pool2d(x, 2)
#         x = self.prelu2_1(self.conv2_1(x))
#         x = self.prelu2_2(self.conv2_2(x))
#         x = F.max_pool2d(x, 2)
#         x = self.prelu3_1(self.conv3_1(x))
#         x = self.prelu3_2(self.conv3_2(x))
#         x = F.max_pool2d(x, 2)
#         x = x.view(-1, 128 * 3 * 3)
#         ip1 = self.preluip1(self.ip1(x))
#         ip2 = self.ip2(ip1)
#         return ip1, ip2

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

        features3d = self.fc1(x)
        # features2d = self.fc2(features3d)
        x = F.prelu(features3d, self.prelu_weight)

        x = self.fc3(x)
        
        return features3d, x



class LMCL_loss(nn.Module):
    """
        Refer to paper:
        Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu
        CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, num_classes, feat_dim, s=7.00, m=0.2):
        super(LMCL_loss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat.cpu(), torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cpu()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits

      
      
# def visualize(feat, labels, epoch):
#     plt.ion()
#     c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
#          '#ff00ff', '#990000', '#999900', '#009900', '#009999']
#     plt.clf()
#     for i in range(10):
#         plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
#     plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
#     #   plt.xlim(xmin=-5,xmax=5)
#     #   plt.ylim(ymin=-5,ymax=5)
#     plt.text(-4.8, 4.6, "epoch=%d" % epoch)
#     plt.savefig('./images/LMCL_loss_u_epoch=%d.jpg' % epoch)
#     # plt.draw()
#     # plt.pause(0.001)
#     plt.close()

def test(test_loder, criterion, model, use_cuda):
    correct = 0
    total = 0
    for i, (data, target) in enumerate(test_loder):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target).cpu()

        feats, _ = model(data)
        logits, mlogits = criterion[1](feats, target)
        _, predicted = torch.max(logits.data, 1)
        total += target.size(0)
        correct += (predicted == target.data).sum()

    print('Test Accuracy of the model on the 10000 test images: %f %%' % (100 * correct / total))


def train(train_loader, model, criterion, optimizer, epoch, loss_weight, use_cuda):
    ip1_loader = []
    idx_loader = []
    for i, (data, target) in enumerate(train_loader):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target).cpu()

        feats, _ = model(data)
        logits, mlogits = criterion[1](feats, target)
        # cross_entropy = criterion[0](logits, target)
        loss = criterion[0](mlogits, target)

        _, predicted = torch.max(logits.data, 1)
        accuracy = (target.data == predicted).float().mean()

        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        loss.backward()

        optimizer[0].step()
        optimizer[1].step()

        ip1_loader.append(feats)
        idx_loader.append((target))
        if (i + 1) % 50 == 0:
            print('Epoch [%d], Iter [%d/%d] Loss: %.4f Acc %.4f'
                  % (epoch, i + 1, len(train_loader), loss.data[0], accuracy))

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
#     visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)


if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False
# Dataset
trainset = datasets.MNIST('./data/', download=True, train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]))
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

testset = datasets.MNIST('./data/', download=True, train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]))
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Model
model = Net()

# NLLLoss
nllloss = nn.CrossEntropyLoss()
# CenterLoss
loss_weight = 0.1
lmcl_loss = LMCL_loss(num_classes=10, feat_dim=FEATURES_DIM)
if use_cuda:
    nllloss = nllloss.cuda()
    # coco_loss = lmcl_loss.cuda()
    model = model.cuda()
criterion = [nllloss, lmcl_loss]
# optimzer4nn
optimizer4nn = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
sheduler_4nn = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.5)

# optimzer4center
optimzer4center = optim.SGD(lmcl_loss.parameters(), lr=0.01)
sheduler_4center = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.5)
for epoch in range(20):
    sheduler_4nn.step()
    sheduler_4center.step()
    # print optimizer4nn.param_groups[0]['lr']
    train(train_loader, model, criterion, [optimizer4nn, optimzer4center], epoch + 1, loss_weight, use_cuda)
    test(test_loader, criterion, model, use_cuda)
    
torch.save(model.state_dict(),"mnist_cnn-cosface.pt")        
torch.save(lmcl_loss.state_dict(),"mnist_loss-cosface.pt")        

