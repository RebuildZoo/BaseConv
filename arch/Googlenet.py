'''
Adapt from Anaconda3\envs\point_learning\lib\site-packages\torchvision\models\googlenet.py
import torchvision
torchvision.models.googLeNet
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2019)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

