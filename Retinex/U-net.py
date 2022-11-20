import torch
from torch import nn
from torch.nn import functional as F

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.Conv2d(out_channel, out_channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.MaxPool2d(kernel_size=3, ceil_mode=True),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, 3, 2, 1)
    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)

class Fusion(nn.Module):

    def __int__(self, in_channel, out_channel, out):
        super(Fusion, self).__int__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 2, 1),
            nn.Conv2d(in_channel, out_channel, 3, 2, 1)
        )

    def forwar(self,x):
        return self.layer(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(3, 32)
        self.c2 = Conv_Block(32, 32)
        self.d1 = DownSample(32, 32)
        self.d2 = DownSample(32, 64)
        self.d3 = DownSample(64, 128)
        self.d4 = DownSample(128, 256)
        self.c3 = Conv_Block(256, 512)
        self.c4 = Conv_Block(512, 512)
        self.u1 = UpSample(512, 256)
        self.f1 = Fusion(384, 256)
        self.u2 = UpSample(256, 128)
        self.f2 = Fusion(192, 128)
        self.u3 = UpSample(128, 64)
        self.f3 = Fusion(96, 64)
        self.u4 = UpSample(64, 32)
        self.f4 = Fusion(64, 32)
        self.out = nn.Conv2d(32, 3, 3, 1, 1)


    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.out(O4)

if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    net = UNet()
    print(net(x).shape)