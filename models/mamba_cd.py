import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from models.changemamba_cd_help.Mamba_backbone import Backbone_VSSM
from models.changemamba_cd_help.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from models.changemamba_cd_help.ChangeDecoder import ChangeDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count


class STMambaBCD(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(STMambaBCD, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
    # def forward(self, x):
    #     pre_data = x.clone()
    #     post_data = x.clone()
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        output = self.decoder(pre_features, post_features)

        output = self.main_clf(output)
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        return output

if __name__ == "__main__":
    from thop import profile
    device = "cuda:5"
    # path = r'/data/zht/Pycharmweight/vmamba/vssm_tiny_0230_ckpt_epoch_262.pth'
    cd_model = STMambaBCD(pretrained="", patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], dims=96,
                          ssm_d_state=16, ssm_ratio=2.0, ssm_rank_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
                          ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2",
                          mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, drop_path_rate=0.1, patch_norm=True,
                          norm_layer='ln', downsample_version="v2", patchembed_version="v2", gmlp=False,
                          use_checkpoint=False, device=device).to(device)
    img = torch.randn(1, 3, 256, 256).to(device)
    # print(cd_model)
    # res = cd_model(img, img)
    # print(res.shape)
    flops1, params1 = profile(cd_model, inputs=(img,))
    print("flops=G", flops1 / (1000 ** 3))
    print("parms=M", params1 / (1000 ** 2))
    param_num = 0
    for name, param in cd_model.named_parameters():
        param_num += param.numel()
    print(f"模型的参数量为: {param_num / (1000 ** 2)}")
