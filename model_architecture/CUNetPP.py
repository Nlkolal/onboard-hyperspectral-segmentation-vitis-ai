import torch, torch.nn as nn

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


class CUNetPP(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)

        self.enc_conv_in = nn.Conv2d(in_ch, 8, 3, 1, 1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.dw1 = DWSeparableConv3x3(8, 16)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.dw2 = DWSeparableConv3x3(16, 32)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.up1 = nn.ConvTranspose2d(32, 32, 2, 2)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.up3 = nn.ConvTranspose2d(16, 8, 2, 2)

        self.dec_conv = nn.Conv2d(8, 8, 3, 1, 1, bias=False)
        self.out_conv = nn.Conv2d(8, n_classes, 1)

    def forward(self, x):
        x = self.relu(self.enc_conv_in(x)); x = self.pool1(x)
        x = self.dw1(x); x = self.pool2(x)
        x = self.dw2(x); x = self.pool3(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.relu(self.dec_conv(x))
        return self.out_conv(x)
