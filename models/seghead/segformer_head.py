# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
from abc import ABCMeta, abstractmethod
import torch.nn as nn
import torch
# from mmcv.cnn import normal_init
# from mmcv.runner import auto_fp16
from mmseg.structures import build_pixel_sampler
from mmcv.cnn import ConvModule
from mmseg.models.utils import resize

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    # @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output



class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# class SegFormerHead(BaseDecodeHead):
#     """
#     SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
#     """
#     def __init__(self, feature_strides, **kwargs):
#         super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
#         assert len(feature_strides) == len(self.in_channels)
#         assert min(feature_strides) == feature_strides[0]
#         self.feature_strides = feature_strides
#
#         c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
#
#         decoder_params = kwargs['decoder_params']
#         embedding_dim = decoder_params['embed_dim']
#
#         self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
#         self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
#         self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
#         self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
#
#         self.linear_fuse = ConvModule(
#             in_channels=embedding_dim*4,
#             out_channels=embedding_dim,
#             kernel_size=1,
#             norm_cfg=dict(type='BN', requires_grad=True)
#         )
#
#         self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
#
#     def forward(self, inputs):
#         x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
#         c1, c2, c3, c4 = x
#
#         ############## MLP decoder on C1-C4 ###########
#         n, _, h, w = c4.shape
#
#         _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
#         _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)
#
#         _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
#         _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)
#
#         _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
#         _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
#
#         _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
#
#         _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
#
#         x = self.dropout(_c)
#         x = self.linear_pred(x)
#
#         return x

class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        # self.defconv_c4 = DeformConv2d(inc=c4_in_channels, outc=embedding_dim, kernel_size=1, padding=0)
        # self.bn_c4 = nn.BatchNorm2d(embedding_dim)
        # self.relu_c4 = nn.ReLU()
        #
        # self.defconv_c3 = DeformConv2d(inc=c3_in_channels, outc=embedding_dim, kernel_size=1, padding=0)
        # self.bn_c3 = nn.BatchNorm2d(embedding_dim)
        # self.relu_c3 = nn.ReLU()
        #
        # self.defconv_c2 = DeformConv2d(inc=c2_in_channels, outc=embedding_dim, kernel_size=1, padding=0)
        # self.bn_c2 = nn.BatchNorm2d(embedding_dim)
        # self.relu_c2 = nn.ReLU()
        #
        # self.defconv_c1 = DeformConv2d(inc=c1_in_channels, outc=embedding_dim, kernel_size=1, padding=0)
        # self.bn_c1 = nn.BatchNorm2d(embedding_dim)
        # self.relu_c1 = nn.ReLU()

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        # print("x0 shape is", x[0].shape)
        c1, c2, c3, c4 = x

        """
        _c4 shape is torch.Size([8, 768, 8, 8])
        _c4 resize shape is torch.Size([8, 768, 64, 64])
        _c3 shape is torch.Size([8, 768, 16, 16])
        _c3 resize shape is torch.Size([8, 768, 64, 64])
        _c2 shape is torch.Size([8, 768, 32, 32])
        _c2 resize shape is torch.Size([8, 768, 64, 64])
        _c1 shape is torch.Size([8, 768, 64, 64])
        """

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        # _c4 = self.defconv_c4(c4)
        # _c4 = self.bn_c4(_c4)
        # _c4 = self.relu_c4(_c4)
        # print("_c4 shape is", _c4.shape)
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)
        # print("_c4 resize shape is", _c4.shape)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        # _c3 = self.defconv_c3(c3)
        # _c3 = self.bn_c3(_c3)
        # _c3 = self.relu_c3(_c3)
        # print("_c3 shape is", _c3.shape)
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)
        # print("_c3 resize shape is", _c3.shape)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        # _c2 = self.defconv_c2(c2)
        # _c2 = self.bn_c2(_c2)
        # _c2 = self.relu_c2(_c2)
        # print("_c2 shape is", _c2.shape)
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
        # print("_c2 resize shape is", _c2.shape)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        # _c1 = self.defconv_c1(c1)
        # _c1 = self.bn_c1(_c1)
        # _c1 = self.relu_c1(_c1)
        # print("_c1 shape is", _c1.shape)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        # print("_c shape is", _c.shape)

        x = self.dropout(_c)
        x = self.linear_pred(x)
        # print("x shape is", x.shape)

        return x

class Fusion(nn.Module):
    def __init__(self, embedding_dim=768):
        super(Fusion, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)

    def forward(self, x_u, x_d):

        ca_score = CA_Block(channel=x_d.shape[1], h=x_d.shape[2], w=x_d.shape[3]).cuda()
        # score = self.sigmoid(x_u + x_d)
        score = ca_score(x_d + x_u)
        x_u_ = x_u * score
        x_d_ = x_d * (1 - score)

        fusion = self.conv(x_u_ + x_d_)

        return fusion

class SegFormerHeadz(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHeadz, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.fusion3 = Fusion(embedding_dim)
        self.fusion2 = Fusion(embedding_dim)
        self.fusion1 = Fusion(embedding_dim)


        # self.linear_fuse = ConvModule(
        #     in_channels=embedding_dim*4,
        #     out_channels=embedding_dim,
        #     kernel_size=1,
        #     norm_cfg=dict(type='BN', requires_grad=True)
        # )
        self.convfusion = nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=7, padding=7 // 2, stride=1,bias=False)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        # print("x0 shape is", x[0].shape)
        c1, c2, c3, c4 = x

        """
        _c4 shape is torch.Size([8, 768, 8, 8])
        _c4 resize shape is torch.Size([8, 768, 64, 64])
        _c3 shape is torch.Size([8, 768, 16, 16])
        _c3 resize shape is torch.Size([8, 768, 64, 64])
        _c2 shape is torch.Size([8, 768, 32, 32])
        _c2 resize shape is torch.Size([8, 768, 64, 64])
        _c1 shape is torch.Size([8, 768, 64, 64])
        """

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        # _c4 = self.defconv_c4(c4)
        # _c4 = self.bn_c4(_c4)
        # _c4 = self.relu_c4(_c4)
        # print("_c4 shape is", _c4.shape)
        _c4 = resize(_c4, size=c3.size()[2:],mode='bilinear',align_corners=False)
        # print("_c4 resize shape is", _c4.shape)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        # _c3 = self.defconv_c3(c3)
        # _c3 = self.bn_c3(_c3)
        # _c3 = self.relu_c3(_c3)

        _c3 = self.fusion3(_c3, _c4)
        # print("_c3 shape is", _c3.shape)
        _c3 = resize(_c3, size=c2.size()[2:],mode='bilinear',align_corners=False)
        # print("_c3 resize shape is", _c3.shape)




        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        # _c2 = self.defconv_c2(c2)
        # _c2 = self.bn_c2(_c2)
        # _c2 = self.relu_c2(_c2)

        _c2 = self.fusion2(_c2, _c3)
        # print("_c2 shape is", _c2.shape)
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
        # print("_c2 resize shape is", _c2.shape)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        # _c1 = self.defconv_c1(c1)
        # _c1 = self.bn_c1(_c1)
        # _c1 = self.relu_c1(_c1)
        _c1 = self.fusion1(_c1, _c2)
        # print("_c1 shape is", _c1.shape)

        # _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.convfusion(_c1)
        _c = self.bn(_c)
        # print("_c shape is", _c.shape)

        x = self.dropout(_c)
        x = self.linear_pred(x)
        # print("x shape is", x.shape)
        return x

class Fusion2(nn.Module):
    def __init__(self, embedding_dim=768):
        super(Fusion2, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.conv = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

    def forward(self, x_u, x_d):

        score = self.sigmoid(x_u + x_d)

        x_u_ = x_u * score
        x_d_ = x_d * (1 - score)

        fusion = self.conv(x_u_ + x_d_)

        return fusion

class SegFormerHead2(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead2, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim) #[32, 64, 160, 256]
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.fusion3 = Fusion2(embedding_dim)
        self.fusion2 = Fusion2(embedding_dim)
        self.fusion1 = Fusion2(embedding_dim)


        self.linear_fuse = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        # print("x0 shape is", x[0].shape)
        c1, c2, c3, c4 = x

        """
        _c4 shape is torch.Size([8, 768, 8, 8])
        _c4 resize shape is torch.Size([8, 768, 64, 64])
        _c3 shape is torch.Size([8, 768, 16, 16])
        _c3 resize shape is torch.Size([8, 768, 64, 64])
        _c2 shape is torch.Size([8, 768, 32, 32])
        _c2 resize shape is torch.Size([8, 768, 64, 64])
        _c1 shape is torch.Size([8, 768, 64, 64])
        """

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c3.size()[2:],mode='bilinear',align_corners=False)
        # print("_c4 shape is", _c4.shape)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        # print("c3 shape is", c3.shape)
        _c3 = self.fusion3(_c3, _c4)


        _c3 = resize(_c3, size=c2.size()[2:],mode='bilinear',align_corners=False)
        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = self.fusion2(_c2, _c3)


        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = self.fusion1(_c1, _c2)


        # _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.linear_fuse(_c1)

        x = self.dropout(_c)
        x = self.linear_pred(x)
        # print("x shape is", x.shape)
        return x

if __name__ == "__main__":
    head = SegFormerHeadz(
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        align_corners=False,
        decoder_params=dict({"embed_dim": 768}))

    img_list = []
    img1 = torch.randn([8, 32, 64, 64])
    img2 = torch.randn([8, 64, 32, 32])
    img3 = torch.randn([8, 160, 16, 16])
    img4 = torch.randn([8, 256, 8, 8])

    img_list.append(img1)
    img_list.append(img2)
    img_list.append(img3)
    img_list.append(img4)

    res = head(img_list)
    print(res.shape)




