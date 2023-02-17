import torch

import numpy as np
import torch.utils
from PIL import  Image
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

'''https://github.com/wangzihanggg/my_unet/blob/main/unet.py'''
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
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
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
        # nn.BatchNorm2d(out_channel),
        # nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
        # nn.BatchNorm2d(out_channel),
        nn.MaxPool2d(kernel_size=2, stride=2)
        # nn.BatchNorm2d(out_channel),
        # nn.LeakyReLU()
        )
    def forward(self, x):
        # x_t_1 = self.conv1(x)
        # x_t_2 = self.conv2(x_t_1)
        # x_t_3 = self.mp(x_t_2)
        return self.layer(x)
        # return x_t_3


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, 3, 1, 5)
    def forward(self, x):
        # up = F.interpolate(x, scale_factor=2, mode='nearest')
        # out = self.layer(up)
        # return torch.cat((out, feature_map), dim=1)
        out = self.layer(x)
        return out

class UpSample1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample1, self).__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, 3, 1, 9)
    def forward(self, x):
        # up = F.interpolate(x, scale_factor=2, mode='nearest')
        # out = self.layer(up)
        # return torch.cat((out, feature_map), dim=1)
        out = self.layer(x)
        return out

class UpSample2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample2, self).__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, 3, 1, 17)
    def forward(self, x):
        # up = F.interpolate(x, scale_factor=2, mode='nearest')
        # out = self.layer(up)
        # return torch.cat((out, feature_map), dim=1)
        out = self.layer(x)
        return out

class UpSample3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample3, self).__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, 3, 1, 33)
    def forward(self, x):
        # up = F.interpolate(x, scale_factor=2, mode='nearest')
        # out = self.layer(up)
        # return torch.cat((out, feature_map), dim=1)
        out = self.layer(x)
        return out

class Fusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Fusion, self).__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.Conv2d(in_channel, out_channel, 1, 1)
        )

    def forward(self, x, feature_map):
        t = torch.cat((x, feature_map), dim=1)
        x = self.layer1(t)
        return x

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
        self.u2 = UpSample1(256, 128)
        self.f2 = Fusion(192, 128)
        self.u3 = UpSample2(128, 64)
        self.f3 = Fusion(96, 64)
        self.u4 = UpSample3(64, 32)
        self.f4 = Fusion(64, 32)
        self.c5 = Conv_Block(32, 3)


    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.d1(x2)
        x4 = self.d2(x3)
        x5 = self.d3(x4)
        x6 = self.d4(x5)
        x7 = self.c3(x6)
        x8 = self.c4(x7)
        x9 = self.u1(x8)
        x10 = self.f1(x9, x5)
        x11 = self.u2(x10)
        x12 = self.f2(x11, x4)
        x13 = self.u3(x12)
        x14 = self.f3(x13, x3)
        x15 = self.u4(x14)
        x16 = self.f4(x15, x2)
        out = self.c5(x16)
        return out

class Network(nn.Module):
    def  __init__(self):
        super(Network, self).__init__()
        self.stage = 1
        self.unet = UNet()
        self._criterion = nn.MSELoss()

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        ilist, rlist, inlist = [], [], []
        input_op = input
        for i in range(self.stage):
            inlist.append(input_op)
            i = self.unet(input_op)
            r = input / i
            r = torch.clamp(r, 0, 255)
            input_op = input + r
            ilist.append(i)
            rlist.append(r)

        return ilist, rlist, inlist
    def _loss(self, input):
        i_list, en_list, in_list = self(input)
        loss = 0
        for i in range(self.stage):
            loss += self._criterion(in_list[i], i_list[i])
        return loss



    # x = torch.randn(1, 3, 600, 400)
    # net = UNet()
    # print(net(x).shape)

class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.unet = UNet()


        base_weights = torch.load(weights)
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        i = self.unet(input)
        r = input / i
        r = torch.clamp(r, 0, 255)
        return i, r


    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, i)
        return loss


