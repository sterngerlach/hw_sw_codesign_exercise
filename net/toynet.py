# coding: utf-8
# toynet.py

import torch.nn as nn
import torch.nn.functional as F

class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(
            1, 6, kernel_size=5, stride=1, padding=2, bias=False)
        self.max_pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn0 = nn.BatchNorm2d(num_features=6)
        self.conv1 = nn.Conv2d(
            6, 16, kernel_size=5, stride=1, padding=0, bias=False)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.flatten = nn.Flatten()
        self.linear0 = nn.Linear(in_features=400, out_features=120)
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.linear2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.max_pool0(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.linear0(x)
        x = F.relu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        out = F.log_softmax(x, dim=1)
        return out
