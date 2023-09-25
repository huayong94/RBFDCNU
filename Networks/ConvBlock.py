from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class ConvLayer(nn.Module):
    """
    a combination of conv and relu
    """
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.layer = nn.Sequential(
            OrderedDict([('conv',
                          nn.Conv2d(in_dim,
                                    out_dim,
                                    kernel_size,
                                    stride=stride,
                                    padding=padding)),
                         ('relu', nn.LeakyReLU(0.2, inplace=True))]))

    def forward(self, feature):
        return self.layer(feature)

class ConvBlock(nn.Module):
    """
    Combination of convs of same output dim.
    """
    def __init__(self, num, dim, input_dim):
        """
        Args:
            num (int): the num of convs
            dim (int): the output of all convs
            input_dim (int): the input dim
        """
        super(ConvBlock, self).__init__()
        layers = [('convlayer%d' % (i + 1),
                   ConvLayer(dim if i else input_dim, dim))
                  for i in range(num)]
        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, features):
        return self.block(features)


class InterpolationLayer(nn.Module):
    def __init__(self):
        super(InterpolationLayer, self).__init__()

    def forward(self, fm, cp_loc, scale):
        
        B, C, H, W = fm.shape

        loc = (cp_loc + 1) / scale - 1
        loc[:, :, 0] = 2 * loc[:, :, 0] /  (W - 1) - 1
        loc[:, :, 1] = 2 * loc[:, :, 1] /  (H - 1) - 1
        loc = loc.unsqueeze(2)
        return F.grid_sample(fm, loc, align_corners=True).squeeze(3)


class Transform(nn.Module):
    def __init__(self, inputDim):
        super(Transform, self).__init__()

        self.conv1 = nn.Conv1d(inputDim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, inputDim * inputDim)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.inputDim = inputDim
    
    def forward(self, feature):
        B = feature.shape[0]

        tran = self.relu(self.bn1(self.conv1(feature)))
        tran = self.relu(self.bn2(self.conv2(tran)))
        tran = self.relu(self.bn3(self.conv3(tran)))
        tran = torch.max(tran, 2, keepdim=True)[0]
        tran = tran.view(-1, 1024)

        tran = self.relu(self.bn4(self.fc1(tran)))
        tran = self.relu(self.bn5(self.fc2(tran)))
        tran = self.fc3(tran)

        iden = Variable(torch.from_numpy(np.eye(self.inputDim).flatten().astype(np.float32))).view(1, self.inputDim * self.inputDim).repeat(B, 1)
        if feature.is_cuda:
            iden = iden.cuda()
        
        tran = tran + iden
        tran = tran.view(-1, self.inputDim, self.inputDim)
        
        feature = feature.transpose(2, 1)
        feature = torch.bmm(feature, tran)
        feature = feature.transpose(2, 1)

        return feature


class FeatureEncoder(nn.Module):
    def __init__(self, inputDim):
        super(FeatureEncoder, self).__init__()
        self.transform = Transform(inputDim)
        self.conv1 = nn.Conv1d(inputDim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()

    def forward(self, x):

        xt = self.transform(x)
        x1 = self.relu(self.bn1(self.conv1(xt)))   # 64
        x2 = self.relu(self.bn2(self.conv2(x1)))   # 128
        x3 = self.bn3(self.conv3(x2))              # 1024
        globalFeature = torch.max(x3, 2, keepdim=True)[0]
        globalFeature = globalFeature.view(-1, 1024, 1).repeat(1, 1, x.shape[2])
        feature = torch.cat([x1, globalFeature], dim=1)
        return feature


class PointEncoder(nn.Module):
    def __init__(self, inputDim):
        super(PointEncoder, self).__init__()
        
        self.transform1 = Transform(inputDim)
        self.conv1 = nn.Conv1d(inputDim, 64, 1)
        self.transform2 = Transform(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = x.transpose(2, 1)
        xt1 = self.transform1(x)
        x1 = self.relu(self.bn1(self.conv1(xt1)))
        xt2 = self.transform2(x1)
        x2 = self.relu(self.bn2(self.conv2(xt2)))
        x3 = self.relu(self.bn3(self.conv3(x2)))
        globalFeature = torch.max(x3, 2, keepdim=True)[0]
        globalFeature = globalFeature.view(-1, 1024, 1).repeat(1, 1, x.shape[2])
        feature = torch.cat([x1, globalFeature], dim=1)
        return feature