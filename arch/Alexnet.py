'''

Adapt from Anaconda3\envs\point_learning\lib\site-packages\torchvision\models\alexnet.py
import torchvision
torchvision.models.alexnet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2019)

class AlexNet_cifar(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(AlexNet_cifar, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNet_mnist(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(AlexNet_mnist, self).__init__()
        # input size: 28 x 28
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5), # o = 24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # o = 11

            nn.Conv2d(16, 64, kernel_size=3, padding=1), # o = 11
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # o = 5

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # o= 5
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1), # o= 5
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1), # o = 5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1), # o = 3
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    
    in_Tsor = torch.randn(5, 1, 28, 28)

    t_net = AlexNet_mnist()

    out_Tsor = t_net(in_Tsor)

    print("output tensor shape: ", out_Tsor.shape)
