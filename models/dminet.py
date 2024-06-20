import torch
import torch.nn as nn
# from .resnet import resnet18
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import models

def draw_features(x, savename, width=8, height=8):
    # tic=time.time()
    fig = plt.figure(figsize=(60, 60))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    # print("time:{}".format(time.time()-tic))

def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class decode(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, out_channel, norm_layer=nn.BatchNorm2d):
        super(decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channel)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CrossAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        self.query2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU())  # conv_f

    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2)
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        q = torch.cat([q1, q2], 1).view(batch_size, -1, height * width).permute(0, 2, 1)
        attn_matrix1 = torch.bmm(q, k1)
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1

        attn_matrix2 = torch.bmm(q, k2)
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2

        feat_sum = self.conv_cat(torch.cat([out1, out2], 1))
        return feat_sum, out1, out2

class DMINet(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(DMINet, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps

        self.resnet = models.resnet18(pretrained=True)
        # self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))
        self.resnet.layer4 = nn.Identity()

        self.cross2 = CrossAtt(256, 256)
        self.cross3 = CrossAtt(128, 128)
        self.cross4 = CrossAtt(64, 64)

        self.Translayer2_1 = BasicConv2d(256, 128, 1)
        self.fam32_1 = decode(128, 128, 128)  # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(128, 64, 1)
        self.fam43_1 = decode(64, 64, 64)  # AlignBlock(64) # decode(64,64,64)

        self.Translayer2_2 = BasicConv2d(256, 128, 1)
        self.fam32_2 = decode(128, 128, 128)
        self.Translayer3_2 = BasicConv2d(128, 64, 1)
        self.fam43_2 = decode(64, 64, 64)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.final = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
        )
        self.final2 = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
        )
        self.final2_2 = nn.Sequential(
            Conv(128, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
        )
        if normal_init:
            self.init_weights()

    # def forward(self, imgs):
    #     imgs1 = imgs.clone()
    #     imgs2 = imgs.clone()
    def forward(self, imgs1, imgs2, labels=None):

        c0 = self.resnet.conv1(imgs1)
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)
        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2)

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3 - cur2_3), self.Translayer2_2(torch.abs(cur1_2 - cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4 - cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)
        out_1 = self.final(out4_up) #H/4 * W/4
        out_2 = self.final2(out4_2_up) #H/4 * W/4
        # out = out_1 + out_2

        out_1_2 = self.final_2(self.upsamplex8(out3))
        out_2_2 = self.final2_2(self.upsamplex8(out3_2))

        if self.show_Feature_Maps:
            savepath = r'temp'
            draw_features(8, 8, (F.interpolate(c1, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),
                          "{}/c1_img1.png".format(savepath))
            draw_features(8, 16, (F.interpolate(c2, scale_factor=8, mode='bilinear')).cpu().detach().numpy(),
                          "{}/c2_img1.png".format(savepath))
            draw_features(16, 16, (F.interpolate(c3, scale_factor=8, mode='bilinear')).cpu().detach().numpy(),
                          "{}/c3_img1.png".format(savepath))
            draw_features(8, 8, (F.interpolate(c1_img2, scale_factor=4, mode='bilinear')).cpu().detach().numpy(),
                          "{}/c1_img2.png".format(savepath))
            draw_features(8, 16, (F.interpolate(c2_img2, scale_factor=8, mode='bilinear')).cpu().detach().numpy(),
                          "{}/c2_img2.png".format(savepath))
            draw_features(16, 16, (F.interpolate(c3_img2, scale_factor=8, mode='bilinear')).cpu().detach().numpy(),
                          "{}/c3_img2.png".format(savepath))
            # You can show more.

        return out_1, out_2, out_1_2, out_2_2

    def init_weights(self):
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final2_2.apply(init_weights)


if __name__ == "__main__":
    from thop import profile
    dminet = DMINet()
    img = torch.rand([8, 3, 256, 256])
    # out_1, out_2, out_1_2, out_2_2 = dminet(img, img)

    input1 = torch.randn(1, 3, 256, 256)
    flops1, params1 = profile(dminet, inputs=(input1,))
    print("flops=G", flops1 / (1000 ** 3))
    print("parms=M", params1 / (1000 ** 2))

