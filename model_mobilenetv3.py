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
    

class SqueezeExcitation(nn.Module): 
    
    def __init__(self, in_channels, reduction = 4):

        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), #Here we're gonna pool it globally into one scalar
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size = 1 ), 
            nn.ReLU(inplace = True), 
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size = 1),
            nn.Sigmoid()
        )

    def forward(self, x): 
        scale = self.se(x)

        return x * scale

class MobileNetBlock(nn.Module): #NEW MobileNetV3 block, kept name the same to prevent errors 
    '''
        Adapt from MobileNet to save computational resources
    '''
    def __init__(self, in_channels, out_channels, expansion = 6, kernel_size = 1,  stride = 2):
        super(MobileNetBlock, self).__init__()

        #expanding, had to use conv2d instead of conv because I didn't want to switch activation in conv incase it didn't work out.
        self.expand = nn.Conv2d(in_channels, in_channels * expansion, kernel_size = 1) if expansion > 1 else None
        self.bn1 = nn.BatchNorm2d(in_channels * expansion) if expansion > 1 else None
        self.act1 = nn.Hardswish()



        self.depthwise = nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size = kernel_size, stride = stride, 
        padding=kernel_size // 2, groups=in_channels * expansion, bias=False) 
        self.bn2 = nn.BatchNorm2d(in_channels * expansion)



        self.se = SqueezeExcitation(in_channels * expansion)

        self.pointwise = nn.Conv2d(in_channels * expansion, out_channels, 1) # Increasing the depth
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        if self.expand: 
            x = self.act1(self.bn1(self.expand(x)))

        
        x = self.bn2(self.depthwise(x))

        x = self.se(x)

        x = self.bn3(self.pointwise(x))

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
        
        self.encoder1 = MobileNetBlock(self.input_channels, 16)
        self.encoder2 = MobileNetBlock(16, 32)
        self.encoder3 = MobileNetBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, self.output_channels)
        
    def forward(self, x):
        # x: N * 3 * H * W
        x = self.encoder1(x) # N * 16 * H/2 * W/2
        x = self.encoder2(x) # N * 32 * H/4 * W/4
        x = self.encoder3(x) # N * 64 * H/8 * W/8
        x = self.pool(x) # N * 64 * 1 * 1
        x = x.view(x.size(0), -1) # N * 64
        x = self.fc(x) # N * (W/8)
        return x
    
    def compute_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
