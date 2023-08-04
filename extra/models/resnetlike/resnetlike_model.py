import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class ResidualBlock(nn.Module):    
  def __init__(self, in_channels, out_channels, stride=1):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding =1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding =1)
    self.bnw= nn.BatchNorm2d(out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)

    if in_channels != out_channels or stride !=1:
      self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels))
    else: 
      self.shortcut = nn.Identity() #jst ret the input as output 

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = F.relu(self.bn1(out))
    out = self.bn2(out)
    out += self.shortcut(residual)
    out = F.relu(out)
    return out

class ResNet(nn.Module):
  def __init__(self):
    super(ResNet, self).__init__()
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.res_block1 = ResidualBlock(32, 64, stride=1)
    self.res_block2 = ResidualBlock(64, 64, stride=2)
    self.fc1 = nn.Linear(64 * 5 * 5, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(self.bn1(x))
    x = self.res_block1(x)
    x = self.res_block2(x)
    x = F.max_pool2d(x, 2)
    x = x.reshape(-1, 64 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    x = F.log_softmax(x, dim=1)
    return x
 