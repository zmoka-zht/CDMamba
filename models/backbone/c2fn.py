import torch
import torch.nn as nn
import torch.nn.functional as F
from models import resnet50
# from .ResNet import ResNet
# import torchvision.models as models

def cus_sample(feat, **kwargs):
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)


def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class MSCA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MSCA, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei


class ACFM(nn.Module):
    def __init__(self, channel=64):
        super(ACFM, self).__init__()

        self.msca = MSCA()
        self.upsample = cus_sample
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

    def forward(self, x, y):
        y = self.upsample(y, scale_factor=2)
        xy = x + y
        wei = self.msca(xy)
        xo = x * wei + y * (1 - wei)
        xo = self.conv(xo)

        return xo


class DGCM(nn.Module):
    def __init__(self, channel=64):
        super(DGCM, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)

        self.h2l = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.h2h = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

        self.mscah = MSCA()
        self.mscal = MSCA()

        self.upsample_add = upsample_add
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

    def forward(self, x):
        # first conv
        x_h = self.h2h(x)
        x_l = self.h2l(self.h2l_pool(x))
        x_h = x_h * self.mscah(x_h)
        x_l = x_l * self.mscal(x_l)
        out = self.upsample_add(x_l, x_h)
        out = self.conv(out)

        return out


class C2FNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64):
        super(C2FNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = resnet50(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)

        self.acfm3 = ACFM()
        self.acfm2 = ACFM()

        self.dgcm3 = DGCM()
        self.dgcm2 = DGCM()

        self.upconv3 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1, relu=True)
        self.upconv2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1, relu=True)

        self.classifier = nn.Conv2d(64, 2, 1)
        self.upsample_add = upsample_add

        # if self.training:
        # self.initialize_weights()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        x2_rfb = self.rfb2_1(x2)  # channel -> 64
        x3_rfb = self.rfb3_1(x3)  # channel -> 64
        x4_rfb = self.rfb4_1(x4)  # channel -> 64

        x43 = self.acfm3(x3_rfb, x4_rfb)
        out43 = self.upconv3(self.dgcm3(x43) + x43)

        x432 = self.acfm2(x2_rfb, out43)
        out432 = self.upconv2(self.dgcm2(x432) + x432)

        s3 = self.classifier(out432)
        s3 = F.interpolate(s3, scale_factor=8, mode='bilinear', align_corners=False)

        return s3

    # def initialize_weights(self):
    # res50 = models.resnet50(pretrained=True)
    # pretrained_dict = res50.state_dict()
    # all_params = {}
    # for k, v in self.resnet.state_dict().items():
    # if k in pretrained_dict.keys():
    # v = pretrained_dict[k]
    # all_params[k] = v
    # assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
    # self.resnet.load_state_dict(all_params)


if __name__ == '__main__':
    ras = C2FNet()
    input_tensor = torch.randn(8, 3, 256, 256)

    out = ras(input_tensor)

    print(out.shape)

    layer = BasicConv2d(in_planes=3, out_planes=2, kernel_size=(1, 3))
    res = layer(input_tensor)
    print(res.shape)