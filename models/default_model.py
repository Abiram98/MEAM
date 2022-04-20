import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6, kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,10,5)
        self.pool2 = nn.AdaptiveMaxPool2d(20)
        self.fc1 = nn.Linear(4000,120) #1360
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,1)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x