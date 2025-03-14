import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        customReLU()
    )
    

class MobileNetBlock(nn.Module):
    '''
        Adapt from MobileNet to save computational resources
    '''
    def __init__(self, in_channels, out_channels, stride = 2):
        super(MobileNetBlock, self).__init__()
        self.depthwise = conv(in_channels, in_channels, 3, stride) # Capturing spatial information
        self.pointwise = conv(in_channels, out_channels, 1) # Increasing the depth
        
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
        self.height = int(config.config["image_height"] / 8)
        self.width = int(config.config["image_width"] / 4)
        
        self.encoder1 = MobileNetBlock(self.input_channels, 4)
        self.encoder2 = MobileNetBlock(4, 8)
        self.encoder3 = MobileNetBlock(8, 16)
        self.pool = nn.MaxPool2d(kernel_size=(self.height, self.width), stride=1)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, self.output_channels)
        
    def forward(self, x):
        # x: N * 3 * H * W
        x = self.encoder1(x) # N * 4 * H/2 * W/2
        # print(f"Encoder1 output shape: {x.shape}")
        x = self.encoder2(x) # N * 8 * H/4 * W/4
        # print(f"Encoder2 output shape: {x.shape}")
        x = self.encoder3(x) # N * 16 * H/8 * W/8
        # print(f"Encoder3 output shape: {x.shape}")
        x = self.pool(x) # N * 16 * H/16 * W/16
        # print(f"Pool output shape: {x.shape}")
        x = x.view(x.size(0), -1) # N * (H/16 * W/16 * 16)
        # print(f"Flatten output shape: {x.shape}")
        x = self.fc(x) # N * W
        # print(f"FC output shape: {x.shape}")
        return x
    
    def compute_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)