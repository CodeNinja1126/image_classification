import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class resModule(nn.Module):
    def __init__(self, channel):
        super(resModule, self).__init__()
        
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(channel, channel//2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channel//2)
        self.conv2 = nn.Conv2d(channel//2, channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.bn3 = nn.BatchNorm2d(channel)
    
    def forward(self, x):
        origin = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.bn3(x + origin)
    
        return x


def resLayer(channel, iter_num):
    resList = [resModule(channel) for _ in range(iter_num)]
    return nn.Sequential(*resList)


class backboneModel(nn.Module):
    def __init__(self, drop_p=0.3):
        super(backboneModel, self).__init__()
        
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=drop_p)
        
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        self.res1 = resLayer(64, 1)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.res2 = resLayer(128, 2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.res3 = resLayer(256, 8)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.res4 = resLayer(512, 8)
        
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.res5 = resLayer(1024, 8)
        
    def forward(self, x):
        x = self.drop(x)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        
        x = self.res1(x)
        
        x = self.drop(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.res2(x)
        
        x = self.drop(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.res3(x)
        
        x = self.drop(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.res4(x)
        
        x = self.drop(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.res5(x)
        
        return x
        

class classficationModel(nn.Module):
    def __init__(self, num_classes = 18):
        super(classficationModel, self).__init__()
        
        self.relu = nn.ReLU()
        
        self.backbone = backboneModel()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        
        return x

