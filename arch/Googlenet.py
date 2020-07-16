'''
Adapt from Anaconda3\envs\point_learning\lib\site-packages\torchvision\models\googlenet.py
import torchvision
torchvision.models.googLeNet
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
torchvision.models.googlenet

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

class Inception(nn.Module):
    __constants__ = ['branch2', 'branch3', 'branch4']
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
            conv_block(ch5x5, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionAux_cifar10(nn.Module):
    '''
    used in training, softmax0 & softmax1
    '''
    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class GoogLeNet_cifar10(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(GoogLeNet_cifar10, self).__init__()

        conv_block = BasicConv2d
        inception_block =Inception
        inception_aux_block = InceptionAux_cifar10

        self.conv1 = conv_block(in_channels, 16, kernel_size=5, padding=2) # o = 28
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True) # o = 14 (ceil_mode, up divide)
        self.conv2 = conv_block(16, 64, kernel_size=1) # o = 14
        self.conv3 = conv_block(64, 128, kernel_size=3, padding=1) # o = 14
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True) # o = 7

        self.inception3a = inception_block(in_channels = 192, ch1x1 =  64, ch3x3red =  96, ch3x3 = 128, ch5x5red = 16, ch5x5 = 32, pool_proj = 32)
        self.inception3b = inception_block(in_channels = 256, ch1x1 = 128, ch3x3red = 128, ch3x3 = 192, ch5x5red = 32, ch5x5 = 96, pool_proj = 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(in_channels = 480, ch1x1 = 192, ch3x3red =  96, ch3x3 = 208, ch5x5red = 16, ch5x5 = 48, pool_proj = 64)
        self.inception4b = inception_block(in_channels = 512, ch1x1 = 160, ch3x3red = 112, ch3x3 = 224, ch5x5red = 24, ch5x5 = 64, pool_proj = 64)
        self.inception4c = inception_block(in_channels = 512, ch1x1 = 128, ch3x3red = 128, ch3x3 = 256, ch5x5red = 24, ch5x5 = 64, pool_proj = 64)
        self.inception4d = inception_block(in_channels = 512, ch1x1 = 112, ch3x3red = 144, ch3x3 = 288, ch5x5red = 32, ch5x5 = 64, pool_proj = 64)
        self.inception4e = inception_block(in_channels = 528, ch1x1 = 256, ch3x3red = 160, ch3x3 = 320, ch5x5red = 32, ch5x5 = 128, pool_proj = 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(in_channels = 832, ch1x1 = 256, ch3x3red = 160, ch3x3 = 320320, ch5x5red = 32, ch5x5 = 128, pool_proj = 128)
        self.inception5b = inception_block(in_channels = 832, ch1x1 = 384, ch3x3red = 192, ch3x3 = 320384, ch5x5red = 48, ch5x5 = 128, pool_proj = 128)

    def forward(self, x):
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)
        return x 

if __name__ == "__main__":
    
    in_Tsor = torch.randn(5, 3, 28, 28)

    t_net = GoogLeNet_cifar10()

    out_Tsor = t_net(in_Tsor)

    print("output tensor shape: ", out_Tsor.shape)

