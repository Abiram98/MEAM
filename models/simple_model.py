import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,4, kernel_size=2)
        self.pool2 = nn.AdaptiveMaxPool2d(20)
        self.fc1 = nn.Linear(1600,120) #1360
        self.fc2 = nn.Linear(120,1)
    def forward(self,x):
        x = self.pool2(F.relu(self.conv1(x)))
        x = torch.flatten(x,1)
        
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x