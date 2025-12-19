import torch, torch.nn as nn
import torch.nn.functional as F

class DWSeparableConv3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, groups=1, bias=False)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(x)
        return x

class CFCN(nn.Module):
    def __init__(self,in_ch, n_classes):
        super().__init__()
        self.dw_1 = DWSeparableConv3x3(in_ch, 10)
        self.dw_2 = DWSeparableConv3x3(10, 20)
        self.dw_3 = DWSeparableConv3x3(20, 40)
        #self.conv1 = nn.Conv2d(40, 40, kernel_size=3, padding=1, bias=False) # For testing
        self.conv_1x1 = nn.Conv2d(in_channels=40, out_channels=n_classes, kernel_size=1)

        self.pool2x2 = nn.MaxPool2d(2, 2)


    def forward(self, x):
        x = self.dw_1(x)
        x = self.pool2x2(x)
        x = self.dw_2(x)
        x = self.pool2x2(x)
        x = self.dw_3(x)
        x = self.conv_1x1(x)
        x = F.interpolate(input=x, scale_factor=4, align_corners=False, mode='bilinear')
        return x



        
