import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import os, sys
import numpy as np

from mpl_toolkits import mplot3d

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

print("Pytorch version:  " + str(torch.__version__))
use_cuda = torch.cuda.is_available()
print("Use CUDA: " + str(use_cuda))

# Cosface
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd.function import Function


from pdb import set_trace as bp

BATCH_SIZE = 100
FEATURES_DIM = 3

BATCH_SIZE_TEST = 1000
EPOCHS = 20
LOG_INTERVAL = 10
 
##### TO CREATE A SERIES OF PICTURES
 
def make_views(ax,angles,elevation=None, width=16, height = 9,
                prefix='tmprot_',**kwargs):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created. 
     
    Returns: the list of files created (for later removal)
    """
     
    files = []
    ax.figure.set_size_inches(width,height)
     
    for i,angle in enumerate(angles):
     
        ax.view_init(elev = elevation, azim=angle)
        fname = '%s%03d.jpeg'%(prefix,i)
        ax.figure.savefig(fname)
        files.append(fname)
     
    return files
 
 
 
##### TO TRANSFORM THE SERIES OF PICTURE INTO AN ANIMATION
 
def make_movie(files,output, fps=10,bitrate=1800,**kwargs):
    """
    Uses mencoder, produces a .mp4/.ogv/... movie from a list of
    picture files.
    """
     
    output_name, output_ext = os.path.splitext(output)
    command = { '.mp4' : 'mencoder "mf://%s" -mf fps=%d -o %s.mp4 -ovc lavc\
                         -lavcopts vcodec=msmpeg4v2:vbitrate=%d'
                         %(",".join(files),fps,output_name,bitrate)}
                          
    command['.ogv'] = command['.mp4'] + '; ffmpeg -i %s.mp4 -r %d %s'%(output_name,fps,output)
     
    print(command[output_ext])
    output_ext = os.path.splitext(output)[1]
    os.system(command[output_ext])
 
 
 
def make_gif(files,output,delay=100, repeat=True,**kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """
     
    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'
              %(delay,loop," ".join(files),output))
 
 
 
 
def make_strip(files,output,**kwargs):
    """
    Uses imageMagick to produce a .jpeg strip from a list of
    picture files.
    """
     
    os.system('montage -tile 1x -geometry +0+0 %s %s'%(" ".join(files),output))
     
     
     
##### MAIN FUNCTION
 
def rotanimate(ax, angles, output, **kwargs):
    """
    Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
    a 3D ax
     
    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """
         
    output_ext = os.path.splitext(output)[1]
 
    files = make_views(ax,angles, **kwargs)
     
    D = { '.mp4' : make_movie,
          '.ogv' : make_movie,
          '.gif': make_gif ,
          '.jpeg': make_strip,
          '.png':make_strip}
           
    D[output_ext](files,output,**kwargs)
     
    for f in files:
        os.remove(f)
     

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
        
class LMCL_loss(nn.Module):
    
    def __init__(self, num_classes, feat_dim, device, s=7.00, m=0.2):
        super(LMCL_loss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.device = device
        self.s_m = s*m

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        feat_l2norm = torch.div(feat, norms)
        feat_l2norm = feat_l2norm * self.s

        norms_w = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        weights_l2norm = torch.div(self.weights, norms_w)
        
        fc7 = torch.matmul(feat_l2norm, torch.transpose(weights_l2norm, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes).to(self.device)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot)
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.s_m)
        output = fc7 - y_onehot

        return output


 
##### EXAMPLE
 
if __name__ == '__main__':
 
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y, Z = axes3d.get_test_data(0.05)
    # s = ax.plot_surface(X, Y, Z, cmap=cm.jet)
    # plt.axis('off') # remove axes for visual appeal
     
    # angles = np.linspace(0,360,21)[:-1] # Take 20 angles between 0 and 360
 
    # # create an animated gif (20ms between frames)
    # rotanimate(ax, angles,'movie.gif',delay=20) 
 
    torch.manual_seed(1)

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




    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net()
    model.eval()
    model.load_state_dict(torch.load("mnist_cnn-cosface.pt", map_location='cpu'))
    model.to(device)

    ind = 142

    image = test_loader.dataset[ind][0].numpy().reshape(28,28)
    lbl = test_loader.dataset[ind][1].numpy()


    image_tensor, label_tensor = test_loader.dataset[ind]
    image_tensor = image_tensor.reshape(1,1,28,28)
    image_tensor, label_tensor = image_tensor.to(device), label_tensor.to(device)

    lmcl_loss = LMCL_loss(num_classes=10, feat_dim=FEATURES_DIM, device=device)
    lmcl_loss.eval()
    lmcl_loss.load_state_dict(torch.load("mnist_loss-cosface.pt", map_location='cpu'))
    lmcl_loss.to(device)

    features3d, pr = model(image_tensor)
    logits = lmcl_loss(features3d, torch.unsqueeze(label_tensor, dim=-1))
    _, prediction = torch.max(logits.data, 1)
    prediction = prediction.cpu().detach().numpy()[0]

    # print ("PREDICTION : " + str(prediction) )

    f3d = []
    # f2d = []
    lbls = []
    for i in range(10000):
        image_tensor, label_tensor = test_loader.dataset[i]
        image_tensor = image_tensor.reshape(1,1,28,28)
        image_tensor, label_tensor = image_tensor.to(device), label_tensor.to(device)

        features3d, pr  = model(image_tensor)
        logits = lmcl_loss(features3d, torch.unsqueeze(label_tensor, dim=-1))
        _, prediction = torch.max(logits.data, 1)

        f3d.append(features3d[0].cpu().detach().numpy())
    #     f2d.append(features2d[0].cpu().detach().numpy())
        
        prediction = prediction.cpu().detach().numpy()[0]
        lbls.append(prediction)

    #     print("features3d:  " + str(features3d[0].detach().numpy()))
    #     print("features2d:  " + str(features2d[0].detach().numpy()))

    # feat3d = np.array(f3d)
    # print("3d features shape" + str(feat3d.shape))

    feat3d = np.array(f3d)
    print("3d features shape" + str(feat3d.shape))

    lbls = np.array(lbls)
    print("labels shape" + str(lbls.shape))



    fig = plt.figure(figsize=(16,9))
    ax = plt.axes(projection='3d')

    for i in range(10):
        # Data for three-dimensional scattered points
        xdata = feat3d[lbls==i,2].flatten()
        ydata = feat3d[lbls==i,0].flatten()
        zdata = feat3d[lbls==i,1].flatten()
        ax.scatter3D(xdata, ydata, zdata);
    ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.show()

    # angles = np.linspace(0,360,21)[:-1] # Take 20 angles between 0 and 360
    angles = np.linspace(0,360,181)[:-1] # Take 20 angles between 0 and 360
 
    # create an animated gif (30ms between frames)
    rotanimate(ax, angles,'movie.gif',delay=10) 
 
