import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import config


class customReLU(nn.Module):
    def forward(self, x):
        return torch.max(x, torch.tensor(0.0, device=config.config["device"]))
    

def conv(in_channels, out_channels, kernel_size, stride=1):
    '''
        Convolutional layer with batch normalization and custom ReLU activation (so that onnx can work)
    '''
    padding = (kernel_size - 1) // 2 # Ensure the same output size as input size
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    

class MobileNetBlock(nn.Module):
    '''
        Adapt from MobileNet to save computational resources
    '''
    def __init__(self, in_channels, out_channels, stride = 2):
        super(MobileNetBlock, self).__init__()
        self.depthwise = conv(in_channels, in_channels, 3, stride) # Capturing spatial information
        self.pointwise = conv(in_channels, out_channels, 1) # Increasing the depth
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ShallowDepthModel(nn.Module):
    '''
        Modified DepthModel with only Encoder to output the depth vector instead of depth map
        Input: N * 3 * H * W (3 = rgb)
        Output: N * W (depth vector of the center line)
    '''
    def __init__(self):
        super(ShallowDepthModel, self).__init__()
        self.input_channels = config.config["input_channels"]
        self.output_channels = config.config["output_channels"]
        
        self.encoder1 = MobileNetBlock(self.input_channels, 12) #CHANGED NUMBER OF FILTERS
        self.encoder2 = MobileNetBlock(12, 24)
        self.encoder3 = MobileNetBlock(24, 48)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p = 0.5) #if needs to be even faster, try reducing to 0.3.  
        self.fc = nn.LazyLinear(self.output_channels)
        
    def forward(self, x):
        x = self.encoder1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.encoder2(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.encoder3(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def compute_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    