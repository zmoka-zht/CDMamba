import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

from models.ifnet_help._block import Conv1x1, make_norm
from models.ifnet_help._common import ChannelAttention, SpatialAttention


class VGG16FeaturePicker(nn.Module):
    def __init__(self, indices=(3,8,15,22,29)):
        super().__init__()
        features = list(vgg16(pretrained=True).features)[:30]
        self.features = nn.ModuleList(features).eval()
        self.indices = set(indices)

    def forward(self, x):
        picked_feats = []
        for idx, model in enumerate(self.features):
            x = model(x)
            if idx in self.indices:
                picked_feats.append(x)
        return picked_feats


def conv2d_bn(in_ch, out_ch, with_dropout=True):
    lst = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.PReLU(),
        make_norm(out_ch),
    ]
    if with_dropout:
        lst.append(nn.Dropout(p=0.6))
    return nn.Sequential(*lst)


class DSIFN(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()

        self.encoder1 = self.encoder2 = VGG16FeaturePicker()

        self.sa1 = SpatialAttention()
        self.sa2= SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()

        self.ca1 = ChannelAttention(in_ch=1024)
        self.bn_ca1 = make_norm(1024)
        self.o1_conv1 = conv2d_bn(1024, 512, use_dropout)
        self.o1_conv2 = conv2d_bn(512, 512, use_dropout)
        self.bn_sa1 = make_norm(512)
        self.o1_conv3 = Conv1x1(512, 2)
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.ca2 = ChannelAttention(in_ch=1536)
        self.bn_ca2 = make_norm(1536)
        self.o2_conv1 = conv2d_bn(1536, 512, use_dropout)
        self.o2_conv2 = conv2d_bn(512, 256, use_dropout)
        self.o2_conv3 = conv2d_bn(256, 256, use_dropout)
        self.bn_sa2 = make_norm(256)
        self.o2_conv4 = Conv1x1(256, 2)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.ca3 = ChannelAttention(in_ch=768)
        self.o3_conv1 = conv2d_bn(768, 256, use_dropout)
        self.o3_conv2 = conv2d_bn(256, 128, use_dropout)
        self.o3_conv3 = conv2d_bn(128, 128, use_dropout)
        self.bn_sa3 = make_norm(128)
        self.o3_conv4 = Conv1x1(128, 2)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        self.ca4 = ChannelAttention(in_ch=384)
        self.o4_conv1 = conv2d_bn(384, 128, use_dropout)
        self.o4_conv2 = conv2d_bn(128, 64, use_dropout)
        self.o4_conv3 = conv2d_bn(64, 64, use_dropout)
        self.bn_sa4 = make_norm(64)
        self.o4_conv4 = Conv1x1(64, 2)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.ca5 = ChannelAttention(in_ch=192)
        self.o5_conv1 = conv2d_bn(192, 64, use_dropout)
        self.o5_conv2 = conv2d_bn(64, 32, use_dropout)
        self.o5_conv3 = conv2d_bn(32, 16, use_dropout)
        self.bn_sa5 = make_norm(16)
        self.o5_conv4 = Conv1x1(16, 2)

    def forward(self, t1, t2, gt):
        # Extract bi-temporal features
        with torch.no_grad():
            self.encoder1.eval(), self.encoder2.eval()
            t1_feats = self.encoder1(t1)
            t2_feats = self.encoder2(t2)

        t1_f_l3, t1_f_l8, t1_f_l15, t1_f_l22, t1_f_l29 = t1_feats
        t2_f_l3, t2_f_l8, t2_f_l15, t2_f_l22, t2_f_l29,= t2_feats

        # Multi-level decoding
        x = torch.cat([t1_f_l29, t2_f_l29], dim=1)
        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)

        out1 = self.o1_conv3(x)

        x = self.trans_conv1(x)
        x = torch.cat([x, t1_f_l22, t2_f_l22], dim=1)
        x = self.ca2(x)*x
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) *x
        x = self.bn_sa2(x)

        out2 = self.o2_conv4(x)

        x = self.trans_conv2(x)
        x = torch.cat([x, t1_f_l15, t2_f_l15], dim=1)
        x = self.ca3(x)*x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) *x
        x = self.bn_sa3(x)

        out3 = self.o3_conv4(x)

        x = self.trans_conv3(x)
        x = torch.cat([x, t1_f_l8, t2_f_l8], dim=1)
        x = self.ca4(x)*x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) *x
        x = self.bn_sa4(x)

        out4 = self.o4_conv4(x)

        x = self.trans_conv4(x)
        x = torch.cat([x, t1_f_l3, t2_f_l3], dim=1)
        x = self.ca5(x)*x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) *x
        x = self.bn_sa5(x)

        out5 = self.o5_conv4(x)

        gt = gt.unsqueeze(1)
        gt5 = gt.squeeze(1).long()
        gt4 = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(1).long()
        gt3 = F.interpolate(gt, scale_factor=0.25, mode='bilinear', align_corners=False).squeeze(1).long()
        gt2 = F.interpolate(gt, scale_factor=0.125, mode='bilinear', align_corners=False).squeeze(1).long()
        gt1 = F.interpolate(gt, scale_factor=0.0625, mode='bilinear', align_corners=False).squeeze(1).long()

        return out5, out4, out3, out2, out1, gt5, gt4, gt3, gt2, gt1

if __name__ == "__main__":
    ifnet = DSIFN()
    img = torch.rand([8, 3, 256, 256])
    # img = torch.rand([8, 3, 512, 512])
    # gt = torch.rand([8, 256, 256])
    # #self._pxl_loss(self.G_pred1, gt) + self._pxl_loss(self.G_pred2, gt) + 0.5*self._pxl_loss(self.G_pred3, gt)
    # out5, out4, out3, out2, out1,gt5, gt4, gt3, gt2, gt1 = ifnet(img, img, gt)
    # print(out5.shape)
    # print(out4.shape)
    # print(out3.shape)
    # print(out2.shape)
    # print(out1.shape)
    # print(gt5.shape)
    # print(gt4.shape)
    # print(gt3.shape)
    # print(gt2.shape)
    # print(gt1.shape)
    from thop import profile

    # caculate flops1
    input1 = torch.randn(1, 3, 256, 256)
    flops1, params1 = profile(ifnet, inputs=(input1,))
    print("flops=G", flops1 / (1000 ** 3))
    print("parms=M", params1 / (1000 ** 2))
