# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F
from models.darnet_help.layers import unetConv2
from models.darnet_help.init_weights import init_weights
from models.darnet_help.layers import ChannelAttention, CDSA, conv_block_nested
import numpy as np

'''
 UNet 3+ STA module Label refinement
'''


class DARNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(DARNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [32, 64, 128, 256, 512]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks  # C = 320

        '''stage 5d'''
        self.sta_d5 = CDSA(filters[4], ds=1)
        self.sta_d5_conv = conv_block_nested(filters[4] * 2, filters[4], filters[4])
        # 求diff
        self.ca_d5 = ChannelAttention(filters[4])

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # Hybrid Attention block
        self.sta_d4 = CDSA(self.CatChannels * 4)
        self.sta_d4_conv = conv_block_nested(self.CatChannels * 8, self.CatChannels * 4, self.CatChannels * 4)
        # 求差diff后concat下层decoder
        self.ca_d4 = ChannelAttention(self.UpChannels)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # Hybrid Attention block
        self.sta_d3 = CDSA(self.CatChannels * 3)
        self.sta_d3_conv = conv_block_nested(self.CatChannels * 6, self.CatChannels * 3, self.CatChannels * 3)
        self.ca_d3 = ChannelAttention(self.UpChannels)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # Hybrid Attention block
        self.sta_d2 = CDSA(self.CatChannels * 2)
        self.sta_d2_conv = conv_block_nested(self.CatChannels * 4, self.CatChannels * 2, self.CatChannels * 2)
        self.ca_d2 = ChannelAttention(self.UpChannels)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        self.sta_d1 = CDSA(self.CatChannels * 1)
        self.sta_d1_conv = conv_block_nested(self.CatChannels * 2, self.CatChannels * 1, self.CatChannels * 1)
        # 求差diff后concat下层decoder
        self.ca_d1 = ChannelAttention(self.UpChannels)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        # self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # Label refinement
        self.lr5_outconv = nn.Conv2d(filters[4], n_classes, 3, padding=1)
        # conv_block_nested(filters[4],n_classes,n_classes)
        self.lr5_out_upsample = nn.Upsample(scale_factor=16, mode='bilinear')

        self.lr4_in_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.lr4_outconv = nn.Conv2d(self.UpChannels + n_classes, n_classes, 3, padding=1)
        # conv_block_nested(self.UpChannels + n_classes,n_classes,n_classes)
        self.lr4_out_upsample = nn.Upsample(scale_factor=8, mode='bilinear')

        self.lr3_in_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.lr3_outconv = nn.Conv2d(self.UpChannels + n_classes, n_classes, 3, padding=1)
        # conv_block_nested(self.UpChannels + n_classes,n_classes,n_classes)
        self.lr3_out_upsample = nn.Upsample(scale_factor=4, mode='bilinear')

        self.lr2_in_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.lr2_outconv = nn.Conv2d(self.UpChannels + n_classes, n_classes, 3, padding=1)
        # conv_block_nested(self.UpChannels + n_classes,n_classes,n_classes)
        self.lr2_out_upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.lr1_in_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.lr1_outconv = nn.Conv2d(self.UpChannels + n_classes, n_classes, 3, padding=1)
        # conv_block_nested(self.UpChannels + n_classes,n_classes,n_classes)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs_t1, inputs_t2):
    # def forward(self, imgs):
    #     inputs_t1 = imgs.clone()
    #     inputs_t2 = imgs.clone()
        ## -------------T1*Encoder-------------
        h1_t1 = self.conv1(inputs_t1)  # h1->320*320*64

        h2_t1 = self.maxpool1(h1_t1)
        h2_t1 = self.conv2(h2_t1)  # h2->160*160*128

        h3_t1 = self.maxpool2(h2_t1)
        h3_t1 = self.conv3(h3_t1)  # h3->80*80*256

        h4_t1 = self.maxpool3(h3_t1)
        h4_t1 = self.conv4(h4_t1)  # h4->40*40*512

        h5_t1 = self.maxpool4(h4_t1)
        hd5_t1 = self.conv5(h5_t1)  # h5->20*20*1024

        ## -------------T2*Encoder-------------
        h1_t2 = self.conv1(inputs_t2)  # h1->320*320*64

        h2_t2 = self.maxpool1(h1_t2)
        h2_t2 = self.conv2(h2_t2)  # h2->160*160*128

        h3_t2 = self.maxpool2(h2_t2)
        h3_t2 = self.conv3(h3_t2)  # h3->80*80*256

        h4_t2 = self.maxpool3(h3_t2)
        h4_t2 = self.conv4(h4_t2)  # h4->40*40*512

        h5_t2 = self.maxpool4(h4_t2)
        hd5_t2 = self.conv5(h5_t2)  # h5->20*20*1024

        ## -------------Decoder-------------

        '''stage 5d'''
        '''Attention Block 5d'''
        hd5_t1_st, hd5_t2_st = self.sta_d5(hd5_t1, hd5_t2)
        hd5_diff = self.sta_d5_conv(torch.cat([hd5_t1_st, hd5_t2_st], dim=1))
        # hd5_diff = hd5_t2_st - hd5_t1_st
        # hd5_diff = hd5_t2 - hd5_t1
        hd5 = self.ca_d5(hd5_diff) * hd5_diff

        '''stage 4d'''
        h1_PT_hd4_t1 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1_t1))))
        h2_PT_hd4_t1 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2_t1))))
        h3_PT_hd4_t1 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3_t1))))
        h4_Cat_hd4_t1 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4_t1)))

        h1_PT_hd4_t2 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1_t2))))
        h2_PT_hd4_t2 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2_t2))))
        h3_PT_hd4_t2 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3_t2))))
        h4_Cat_hd4_t2 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4_t2)))

        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))

        '''Attention Block 4d'''
        s4_t1_cat = torch.cat((h1_PT_hd4_t1, h2_PT_hd4_t1, h3_PT_hd4_t1, h4_Cat_hd4_t1), 1)
        s4_t2_cat = torch.cat((h1_PT_hd4_t2, h2_PT_hd4_t2, h3_PT_hd4_t2, h4_Cat_hd4_t2), 1)
        s4_t1_cat, s4_t2_cat = self.sta_d4(s4_t1_cat, s4_t2_cat)
        # AB_4d_diff = s4_t2_cat - s4_t1_cat
        AB_4d_diff = self.sta_d4_conv(torch.cat([s4_t1_cat, s4_t2_cat], dim=1))
        cat_4d = torch.cat((AB_4d_diff, hd5_UT_hd4), 1)
        AB_4d = self.ca_d4(cat_4d) * cat_4d

        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(AB_4d)))  # hd4->40*40*UpChannels

        '''stage 3d'''
        h1_PT_hd3_t1 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1_t1))))
        h2_PT_hd3_t1 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2_t1))))
        h3_Cat_hd3_t1 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3_t1)))

        h1_PT_hd3_t2 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1_t2))))
        h2_PT_hd3_t2 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2_t2))))
        h3_Cat_hd3_t2 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3_t2)))

        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        '''Attention Block 4d'''
        s3_t1_cat = torch.cat((h1_PT_hd3_t1, h2_PT_hd3_t1, h3_Cat_hd3_t1), 1)
        s3_t2_cat = torch.cat((h1_PT_hd3_t2, h2_PT_hd3_t2, h3_Cat_hd3_t2), 1)
        s3_t1_cat, s3_t2_cat = self.sta_d3(s3_t1_cat, s3_t2_cat)
        # AB_3d_diff = s3_t2_cat - s3_t1_cat
        AB_3d_diff = self.sta_d3_conv(torch.cat([s3_t1_cat, s3_t2_cat], dim=1))
        cat_3d = torch.cat((AB_3d_diff, hd4_UT_hd3, hd5_UT_hd3), 1)
        AB_3d = self.ca_d3(cat_3d) * cat_3d

        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(AB_3d)))  # hd3->80*80*UpChannels

        '''stage 2d'''
        h1_PT_hd2_t1 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1_t1))))
        h2_Cat_hd2_t1 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2_t1)))

        h1_PT_hd2_t2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1_t2))))
        h2_Cat_hd2_t2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2_t2)))

        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))

        '''Attention Block 4d'''
        s2_t1_cat = torch.cat((h1_PT_hd2_t1, h2_Cat_hd2_t1), 1)
        s2_t2_cat = torch.cat((h1_PT_hd2_t2, h2_Cat_hd2_t2), 1)
        s2_t1_cat, s2_t2_cat = self.sta_d2(s2_t1_cat, s2_t2_cat)
        # AB_2d_diff = s2_t2_cat - s2_t1_cat
        AB_2d_diff = self.sta_d2_conv(torch.cat([s2_t1_cat, s2_t2_cat], dim=1))
        cat_2d = torch.cat((AB_2d_diff, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)
        AB_2d = self.ca_d2(cat_2d) * cat_2d

        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(AB_2d)))  # hd2->160*160*UpChannels

        '''stage 1d'''
        h1_Cat_hd1_t1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1_t1)))

        h1_Cat_hd1_t2 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1_t2)))

        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))

        '''Attention Block 4d'''
        s1_t1_cat = h1_Cat_hd1_t1
        s1_t2_cat = h1_Cat_hd1_t2
        # s1_t1_cat,s1_t2_cat= self.sta_d1(s1_t1_cat,s1_t2_cat)
        # AB_1d_diff = s1_t2_cat - s1_t1_cat
        AB_1d_diff = self.sta_d1_conv(torch.cat([s1_t1_cat, s1_t2_cat], dim=1))
        cat_1d = torch.cat((AB_1d_diff, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)
        AB_1d = self.ca_d1(cat_1d) * cat_1d

        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(AB_1d)))  # hd1->320*320*UpChannels

        # d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        # Label refinement
        # hd5~hd1

        d5 = self.lr5_outconv(hd5)
        d5_up = self.lr5_out_upsample(d5)

        d4_up_in = self.lr4_in_upsample(d5)
        d4 = torch.cat([hd4, d4_up_in], dim=1)
        d4 = self.lr4_outconv(d4)
        d4_up = self.lr4_out_upsample(d4)

        d3_up_in = self.lr3_in_upsample(d4)
        d3 = torch.cat([hd3, d3_up_in], dim=1)
        d3 = self.lr3_outconv(d3)
        d3_up = self.lr3_out_upsample(d3)

        d2_up_in = self.lr2_in_upsample(d3)
        d2 = torch.cat([hd2, d2_up_in], dim=1)
        d2 = self.lr2_outconv(d2)
        d2_up = self.lr2_out_upsample(d2)

        d1_up_in = self.lr1_in_upsample(d2)
        d1 = torch.cat([hd1, d1_up_in], dim=1)
        d1 = self.lr1_outconv(d1)

        return d5_up, d4_up, d3_up, d2_up, d1  # (d1,d2_up,d3_up,d4_up,d5_up)
        # return F.sigmoid(d1)

if __name__ == "__main__":
    darnet = DARNet().to("cuda:0")
    img = torch.rand([8, 3, 256, 256]).to("cuda:0")
    #loss_weight [0,2,0.2,0.2,0.2,0.2]
    # res = darnet(img, img)
    # print(res[0].shape)
    # print(res[1].shape)
    # print(res[2].shape)
    # print(res[3].shape)
    # print(res[4].shape)

    input1 = torch.randn(1, 3, 256, 256).to("cuda:0")
    flops1, params1 = profile(darnet, inputs=(input1,))
    print("flops=G", flops1 / (1000 ** 3))
    print("parms=M", params1 / (1000 ** 2))
