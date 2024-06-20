import torch
import torch.nn as nn
from torch.nn import functional as F
# from .base import ASPP
from models.bifa_help.implicit_help import ifa_simfpn
import math
import numpy as np


def get_syncbn():
    return nn.BatchNorm2d
    # return nn.SyncBatchNorm

class ASPP(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, in_planes, inner_planes=256, sync_bn=False, dilations=(12, 24, 36)):
        super(ASPP, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                   padding=dilations[0], dilation=dilations[0], bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                   padding=dilations[1], dilation=dilations[1], bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                   padding=dilations[2], dilation=dilations[2], bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))

        self.out_planes = (len(dilations) + 2) * inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return aspp_out


class fpn_ifa(nn.Module):

    def __init__(self, in_planes, num_classes=2, inner_planes=256, sync_bn=False, dilations=(12, 24, 36),
                 pos_dim=24, ultra_pe=False, unfold=False, no_aspp=False,
                 local=False, stride=1, learn_pe=False, require_grad=False, num_layer=2):

        super(fpn_ifa, self).__init__()
        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.no_aspp = no_aspp

        self.unfold = unfold

        if self.no_aspp:
            self.head = nn.Sequential(nn.Conv2d(in_planes[3], 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True))
        else:
            self.aspp = ASPP(in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)
            self.head = nn.Sequential(
                nn.Conv2d(self.aspp.get_outplanes(), 256, kernel_size=3, padding=1, dilation=1, bias=False),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1))

        self.ifa = ifa_simfpn(ultra_pe=ultra_pe, pos_dim=pos_dim, sync_bn=sync_bn, num_classes=num_classes, local=local,
                              unfold=unfold, stride=stride, learn_pe=learn_pe, require_grad=require_grad,
                              num_layer=num_layer)
        self.enc1 = nn.Sequential(nn.Conv2d(in_planes[0], 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(in_planes[1], 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(in_planes[2], 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x1, x2, x3, x4 = x
        if self.no_aspp:
            aspp_out = self.head(x4)
        else:
            aspp_out = self.aspp(x4)
            aspp_out = self.head(aspp_out)

        x1 = self.enc1(x1)
        x2 = self.enc2(x2)
        x3 = self.enc3(x3)
        context = []
        h, w = x1.shape[-2], x1.shape[-1]

        target_feat = [x1, x2, x3, aspp_out]

        for i, feat in enumerate(target_feat):
            context.append(self.ifa(feat, size=[h, w], level=i + 1))
        context = torch.cat(context, dim=-1).permute(0, 2, 1)


        res = self.ifa(context, size=[h, w], after_cat=True)

        return res