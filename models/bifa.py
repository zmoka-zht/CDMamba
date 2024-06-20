# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------

from functools import partial
from models.seghead.segformer_head import SegFormerHead
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from mmseg.utils import get_root_logger
from mmengine.runner import load_checkpoint
import math
from models.bifa_help.ImplicitFunction import fpn_ifa
from models.backbone.my_transformer import *


class flowmlp(nn.Module):
    def __init__(self, inplane):
        super(flowmlp, self).__init__()
        self.dwconv = nn.Conv2d(inplane*4, inplane*4, 3, 1, 1, bias=True, groups=inplane)
        self.Conv_enlarge = nn.Conv2d(inplane, inplane*4, 1)
        self.Conv_shrink = nn.Conv2d(inplane*4, inplane, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.Conv_enlarge(x)
        x = self.dwconv(x)
        x = self.gelu(x)
        x = self.Conv_shrink(x)

        return x

class DiffFlowN(nn.Module):
    def __init__(self, inplane, h, w):
        """
        implementation of diffflow
        :param inplane:
        :param norm_layer:
        """
        super(DiffFlowN, self).__init__()
        self.flowmlp1 = flowmlp(inplane)
        self.flowmlp2 = flowmlp(inplane)
        self.flow_make1 = nn.Conv2d(inplane *2 , 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x1, x2):

        x1 = self.flowmlp1(x1)
        x2 = self.flowmlp2(x2)

        size = x1.size()[2:]
        flow1 = self.flow_make1(torch.cat([x1, x2], dim=1))


        seg_flow_warp1 = self.flow_warp(x1, flow1, size) #A
        diff1 = torch.abs(seg_flow_warp1 - x2)

        return diff1

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)

        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.cond = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
        self.avgpoolchannel = nn.AdaptiveAvgPool2d(1)


        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, cond):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)


        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class AttentionRealCrossChannel(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.cond = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W, cond):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        kv = self.kv(cond).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).reshape(B, C, N).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.sr_ratio = sr_ratio
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.attn_realchannel = AttentionRealCrossChannel(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, cond):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, cond))
        x = x + self.drop_path(self.attn_realchannel(self.norm1(x), H, W, self.norm1(cond)))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W)) #[B, N, C]

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            # logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features1(self, x, cond):
        B = x.shape[0]

        # stage 1
        x, H, W = self.patch_embed1(x)
        cond, H, W = self.patch_embed1(cond)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W, cond)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_features2(self, x, cond):
        # stage 2
        B = x.shape[0]
        x, H, W = self.patch_embed2(x)
        cond, H, W = self.patch_embed2(cond)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W, cond)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_features3(self, x, cond):
        # stage 3
        B = x.shape[0]
        x, H, W = self.patch_embed3(x)
        cond, H, W = self.patch_embed3(cond)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W, cond)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_features4(self, x, cond):
        # stage 4
        B = x.shape[0]
        x, H, W = self.patch_embed4(x)
        cond, H, W = self.patch_embed4(cond)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W, cond)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def forward(self, x):
        x, outs = self.forward_features1(x)
        x, outs = self.forward_features2(x)
        x, outs = self.forward_features3(x)
        x, outs = self.forward_features4(x)

        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class BiFA(nn.Module):
    def __init__(self, backbone="mit_b5"):
        super().__init__()
        if backbone == "mit_b0":
            self.segformer = mit_b0()
            self.ckpt = torch.load(r"E:\pertrain_weight\segformer\mit_b0.pth")
            self.segformer.load_state_dict(self.ckpt, False)
            self.head = SegFormerHead(in_channels=[32, 64, 160, 256],
                                      in_index=[0, 1, 2, 3],
                                      feature_strides=[4, 8, 16, 32],
                                      channels=128,
                                      dropout_ratio=0.1,
                                      num_classes=2,  # 64, 2
                                      align_corners=False,
                                      decoder_params=dict({"embed_dim": 768}))
        self.ckpt.pop("head.weight")
        self.ckpt.pop("head.bias")
        self.diffflow1 = DiffFlowN(inplane=32, h=64, w=64)
        self.diffflow2 = DiffFlowN(inplane=64, h=32, w=32)
        self.diffflow3 = DiffFlowN(inplane=160, h=16, w=16)
        self.diffflow4 = DiffFlowN(inplane=256, h=8, w=8)
        self.ifa = fpn_ifa(in_planes=256, ultra_pe=True, pos_dim=24, no_aspp=True, require_grad=True)

    def forward(self, x1, x2):
        diff_list = []

        # stage 1
        x1_1 = self.segformer.forward_features1(x1, x2)
        x2_1 = self.segformer.forward_features1(x2, x1)  # [8, 32, 64, 64]

        diff0 = self.diffflow1(x1_1, x2_1)

        # diff0 = torch.abs(x1_1 - x2_1)

        # stage 2
        x1_2 = self.segformer.forward_features2(x1_1, x2_1)
        x2_2 = self.segformer.forward_features2(x2_1, x1_1)

        diff1 = self.diffflow2(x1_2, x2_2)


        # stage 3
        x1_3 = self.segformer.forward_features3(x1_2, x2_2)
        x2_3 = self.segformer.forward_features3(x2_2, x1_2)

        diff2 = self.diffflow3(x1_3, x2_3)
        # print(diff2.shape)

        # stage 4
        x1_4 = self.segformer.forward_features4(x1_3, x2_3)
        x2_4 = self.segformer.forward_features4(x2_3, x1_3)
        diff3 = torch.abs(x1_4 - x2_4)

        diff_list.append(diff0)
        diff_list.append(diff1)
        diff_list.append(diff2)
        diff_list.append(diff3)

        segmap_orign = self.ifa(diff_list)
        return segmap_orign

class BiFA_SCD(nn.Module):
    def __init__(self, backbone="mit_b0", num_classes=7):
        super().__init__()
        if backbone == "mit_b0":
            self.segformer = mit_b0()
            self.ckpt = torch.load(r"/data/zht/Pycharmweight/segformer/mit_b0.pth")
            self.segformer.load_state_dict(self.ckpt, False)
            self.head = SegFormerHead(in_channels=[32, 64, 160, 256],
                                      in_index=[0, 1, 2, 3],
                                      feature_strides=[4, 8, 16, 32],
                                      channels=128,
                                      dropout_ratio=0.1,
                                      num_classes=2,  # 64, 2
                                      align_corners=False,
                                      decoder_params=dict({"embed_dim": 768}))
            channel_list = [32, 64, 160, 256]
        elif backbone == "mit_b3":
            self.segformer = mit_b3()
            self.ckpt = torch.load(r"/data/zht/Pycharmweight/segformer/mit_b3.pth")
            self.segformer.load_state_dict(self.ckpt, False)
            self.head = SegFormerHead(in_channels=[64, 128, 320, 512],
                                      in_index=[0, 1, 2, 3],
                                      feature_strides=[4, 8, 16, 32],
                                      channels=128,
                                      dropout_ratio=0.1,
                                      num_classes=2,
                                      align_corners=False,
                                      decoder_params=dict({"embed_dim": 768}))
            channel_list = [64, 128, 320, 512]
        elif backbone == "mit_b5":
            self.segformer = mit_b5()
            self.ckpt = torch.load(r"/data/zht/Pycharmweight/segformer/mit_b5.pth")
            self.segformer.load_state_dict(self.ckpt, False)
            self.head = SegFormerHead(in_channels=[64, 128, 320, 512],
                                      in_index=[0, 1, 2, 3],
                                      feature_strides=[4, 8, 16, 32],
                                      channels=128,
                                      dropout_ratio=0.1,
                                      num_classes=2,
                                      align_corners=False,
                                      decoder_params=dict({"embed_dim": 768}))
            channel_list = [64, 128, 320, 512]
        # self.ckpt.pop("head.weight")
        # self.ckpt.pop("head.bias")
        self.diffflow1 = DiffFlowN(inplane=channel_list[0], h=64, w=64)
        self.diffflow2 = DiffFlowN(inplane=channel_list[1], h=32, w=32)
        self.diffflow3 = DiffFlowN(inplane=channel_list[2], h=16, w=16)
        self.diffflow4 = DiffFlowN(inplane=channel_list[3], h=8, w=8)
        self.ifa1 = fpn_ifa(in_planes=channel_list, ultra_pe=True, pos_dim=24, no_aspp=True, require_grad=True,
                        num_classes=num_classes)
        self.ifa2 = fpn_ifa(in_planes=channel_list, ultra_pe=True, pos_dim=24, no_aspp=True, require_grad=True,
                        num_classes=num_classes)
        self.unsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        diff_list = []

        # stage 1
        x1_1 = self.segformer.forward_features1(x1, x2)
        x2_1 = self.segformer.forward_features1(x2, x1)  # [8, 32, 64, 64]

        diff0 = self.diffflow1(x1_1, x2_1)

        # diff0 = torch.abs(x1_1 - x2_1)

        # stage 2
        x1_2 = self.segformer.forward_features2(x1_1, x2_1)
        x2_2 = self.segformer.forward_features2(x2_1, x1_1)

        diff1 = self.diffflow2(x1_2, x2_2)


        # stage 3
        x1_3 = self.segformer.forward_features3(x1_2, x2_2)
        x2_3 = self.segformer.forward_features3(x2_2, x1_2)

        diff2 = self.diffflow3(x1_3, x2_3)
        # print(diff2.shape)

        # stage 4
        x1_4 = self.segformer.forward_features4(x1_3, x2_3)
        x2_4 = self.segformer.forward_features4(x2_3, x1_3)
        diff3 = torch.abs(x1_4 - x2_4)

        diff_list.append(diff0)
        diff_list.append(diff1)
        diff_list.append(diff2)
        diff_list.append(diff3)

        segmap_1 = self.ifa1(diff_list)
        segmap_2 = self.ifa2(diff_list)
        segmap_orign1 = self.unsample_4(segmap_1)
        segmap_orign2 = self.unsample_4(segmap_2)
        return segmap_orign1, segmap_orign2

class BiFA_SCD_woADFF(BiFA_SCD):
    def __init__(self, backbone="mit_b0", num_classes=7):
        super(BiFA_SCD_woADFF, self).__init__(backbone=backbone, num_classes=num_classes)

    def forward(self, x1, x2):
        diff_list = []

        # stage 1
        x1_1 = self.segformer.forward_features1(x1, x2)
        x2_1 = self.segformer.forward_features1(x2, x1)  # [8, 32, 64, 64]
        diff0 = torch.abs(x1_1 - x2_1)

        # stage 2
        x1_2 = self.segformer.forward_features2(x1_1, x2_1)
        x2_2 = self.segformer.forward_features2(x2_1, x1_1)
        diff1 = torch.abs(x1_2 - x2_2)

        # stage 3
        x1_3 = self.segformer.forward_features3(x1_2, x2_2)
        x2_3 = self.segformer.forward_features3(x2_2, x1_2)
        diff2 = torch.abs(x1_3 - x2_3)
        # print(diff2.shape)

        # stage 4
        x1_4 = self.segformer.forward_features4(x1_3, x2_3)
        x2_4 = self.segformer.forward_features4(x2_3, x1_3)
        diff3 = torch.abs(x1_4 - x2_4)

        diff_list.append(diff0)
        diff_list.append(diff1)
        diff_list.append(diff2)
        diff_list.append(diff3)

        segmap_1 = self.ifa1(diff_list)
        segmap_2 = self.ifa2(diff_list)
        segmap_orign1 = self.unsample_4(segmap_1)
        segmap_orign2 = self.unsample_4(segmap_2)
        return segmap_orign1, segmap_orign2

class BiFA_SCD_woIFA(BiFA_SCD):
    def __init__(self, backbone="mit_b0", num_classes=7):
        super(BiFA_SCD_woIFA, self).__init__(backbone=backbone, num_classes=num_classes)

        self.head1 = SegFormerHead(in_channels=[64, 128, 320, 512],
                                  in_index=[0, 1, 2, 3],
                                  feature_strides=[4, 8, 16, 32],
                                  channels=128,
                                  dropout_ratio=0.1,
                                  num_classes=num_classes,
                                  align_corners=False,
                                  decoder_params=dict({"embed_dim": 768}))
        self.head2 = SegFormerHead(in_channels=[64, 128, 320, 512],
                                   in_index=[0, 1, 2, 3],
                                   feature_strides=[4, 8, 16, 32],
                                   channels=128,
                                   dropout_ratio=0.1,
                                   num_classes=num_classes,
                                   align_corners=False,
                                   decoder_params=dict({"embed_dim": 768}))

    def forward(self, x1, x2):
        diff_list = []

        # stage 1
        x1_1 = self.segformer.forward_features1(x1, x2)
        x2_1 = self.segformer.forward_features1(x2, x1)  # [8, 32, 64, 64]

        diff0 = self.diffflow1(x1_1, x2_1)

        # diff0 = torch.abs(x1_1 - x2_1)

        # stage 2
        x1_2 = self.segformer.forward_features2(x1_1, x2_1)
        x2_2 = self.segformer.forward_features2(x2_1, x1_1)

        diff1 = self.diffflow2(x1_2, x2_2)


        # stage 3
        x1_3 = self.segformer.forward_features3(x1_2, x2_2)
        x2_3 = self.segformer.forward_features3(x2_2, x1_2)

        diff2 = self.diffflow3(x1_3, x2_3)
        # print(diff2.shape)

        # stage 4
        x1_4 = self.segformer.forward_features4(x1_3, x2_3)
        x2_4 = self.segformer.forward_features4(x2_3, x1_3)
        diff3 = torch.abs(x1_4 - x2_4)

        diff_list.append(diff0)
        diff_list.append(diff1)
        diff_list.append(diff2)
        diff_list.append(diff3)

        segmap_1 = self.head1(diff_list)
        segmap_2 = self.head2(diff_list)
        segmap_orign1 = self.unsample_4(segmap_1)
        segmap_orign2 = self.unsample_4(segmap_2)
        return segmap_orign1, segmap_orign2

if __name__ == "__main__":
    img = torch.randn([6, 3, 512, 512]).to("cuda:0")
    gt = torch.randn([6, 512, 512]).to("cuda:0")
    seg = BiFA_SCD_woIFA(backbone="mit_b3", num_classes=7).to("cuda:0")

    res1, res2 = seg(img, img)
    print("res shape is", res1.shape)
    print("res shape is", res2.shape)


