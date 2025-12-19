import torch, torch.nn as nn

class SmallUNet(nn.Module):
    def __init__(self, in_ch, ncls):
        super().__init__()
        def block(c1,c2):
            return nn.Sequential(nn.Conv2d(c1,c2,3,padding=1), nn.ReLU(True),
                                 nn.Conv2d(c2,c2,3,padding=1), nn.ReLU(True))
        self.d1 = block(in_ch, 32); self.p1 = nn.MaxPool2d(2)
        self.d2 = block(32, 64);    self.p2 = nn.MaxPool2d(2)
        self.b  = block(64,128)
        self.u2 = nn.ConvTranspose2d(128,64,2,2)
        self.u1 = nn.ConvTranspose2d(64,32,2,2)
        self.c2 = block(128,64)
        self.c1 = block(64,32)
        self.out= nn.Conv2d(32, ncls, 1)
    def forward(self,x):
        x1=self.d1(x); x2=self.d2(self.p1(x1)); xb=self.b(self.p2(x2))
        x = self.u2(xb); x = torch.cat([x,x2],1); x=self.c2(x)
        x = self.u1(x);  x = torch.cat([x,x1],1); x=self.c1(x)
        return self.out(x)
