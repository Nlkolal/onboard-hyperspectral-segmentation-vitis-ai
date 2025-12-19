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

class CUNet(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()

        self.relu = nn.ReLU(inplace=False)

        # ---------- Encoder block 1 (3 -> 8) ----------
        self.conv1_1 = nn.Conv2d(in_ch, 8, kernel_size=3, padding=1, bias=False)
        self.conv1_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)
        self.dw1     = DWSeparableConv3x3(8, 8)
        self.pool1   = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------- Encoder block 2 (8 -> 16) ----------
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.dw2     = DWSeparableConv3x3(16, 16)
        self.pool2   = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------- Encoder block 3 (16 -> 32) ----------
        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.dw3     = DWSeparableConv3x3(32, 32)
        self.pool3   = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------- Decoder block 1 (32 -> 32) ----------
        self.up1     = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec1_1  = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.dec1_2  = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)

        # ---------- Decoder block 2 (32 -> 16) ----------
        self.up2     = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2_1  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.dec2_2  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)

        # ---------- Decoder block 3 (16 -> 8) ----------
        self.up3     = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec3_1  = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)
        self.dec3_2  = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)

        # ---------- Final 1x1 classifier ----------
        self.out_conv = nn.Conv2d(8, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        # Encoder
        # Block 1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.dw1(x)
        x = self.pool1(x)

        # Block 2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.dw2(x)
        x = self.pool2(x)

        # Block 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.dw3(x)
        x = self.pool3(x)

        # Decoder
        # Up block 1 (32 -> 32)
        x = self.up1(x)
        x = self.relu(self.dec1_1(x))
        x = self.relu(self.dec1_2(x))

        # Up block 2 (32 -> 16)
        x = self.up2(x)
        x = self.relu(self.dec2_1(x))
        x = self.relu(self.dec2_2(x))

        # Up block 3 (16 -> 8)
        x = self.up3(x)
        x = self.relu(self.dec3_1(x))
        x = self.relu(self.dec3_2(x))

        # Final logits (no sigmoid here)
        x = self.out_conv(x)
        return x


