import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.n3 = nn.LayerNorm(dim)
        self.n4 = nn.LayerNorm(dim)
        self.n5 = nn.LayerNorm(dim)
        self.n6 = nn.LayerNorm(dim)
        self.ln1 = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.ln2 = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.ln3 = nn.Linear(head_dim, 3 * head_dim, bias=qkv_bias)
        self.ln4 = nn.Linear(head_dim, 3 * head_dim, bias=qkv_bias)

        self.ln = nn.Linear(2 * dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape
        l = int(math.sqrt(N))
        self.pos1 = nn.Parameter(torch.randn([1, l, C])).to("cuda:0")
        self.pos2 = nn.Parameter(torch.randn([1, l, C])).to("cuda:0")
        self.pos3 = nn.Parameter(torch.randn([1, self.num_heads, l, C // self.num_heads])).to("cuda:0")
        self.pos4 = nn.Parameter(torch.randn([1, self.num_heads, l, C // self.num_heads])).to("cuda:0")
        x_w = self.n3(x).reshape(B, l, l, C).view(B * l, l, C)  # .transpose(1,2)
        # print(x_w.shape)

        x_w = x_w + self.pos1
        x_w_q, x_w_k, x_w_v = self.ln1(x_w).reshape(B * l, l, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3,
                                                                                                              1,
                                                                                                              4)  # B*l,head,lw,C
        x_ww = ((x_w_q @ x_w_k.transpose(-1, -2)) * self.scale).softmax(-1)  # B*l,head,lw,lw
        x_ww = (x_ww @ x_w_v)  # .transpose(1, 2).reshape(B, N, C) # B,lh,lw,C

        x_ww = x_ww + self.pos3
        x_w_q, x_w_k, x_w_v = self.ln3(x_ww).reshape(B * l, self.num_heads, l, 3, C // self.num_heads).permute(3, 0, 1,
                                                                                                               2, 4)

        x_h = self.n4(x).reshape(B, l, l, C).transpose(1, 2).contiguous().view(B * l, l, C)  # .transpose(1,2)

        x_h = x_h + self.pos2
        x_h_q, x_h_k, x_h_v = self.ln2(x_h).reshape(B * l, l, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3,
                                                                                                              1,
                                                                                                              4)  # B*l,head,lh,C
        x_hh = ((x_h_q @ x_h_k.transpose(-1, -2)) * self.scale).softmax(-1)  # B*l,head,lh,lh
        x_hh = (
                    x_hh @ x_h_v)  # .reshape(B, l, self.num_heads,l, C // self.num_heads).permute(0,3,1,2,4).reshape(B, N, C)

        x_hh = x_hh + self.pos4
        x_h_q, x_h_k, x_h_v = self.ln4(x_hh).reshape(B * l, self.num_heads, l, 3, C // self.num_heads).permute(3, 0, 1,
                                                                                                               2, 4)

        # =========================================
        score1 = ((x_h_q @ x_w_k.transpose(-1, -2)) * self.scale).softmax(-1)
        out1 = (score1 @ x_w_v).transpose(1, 2).reshape(B * l, l, C)  # B*l,lh,C
        out1 = out1.reshape(B, l, l, C).transpose(1, 2)  # B,h,w,C
        # =========================================
        score2 = ((x_w_q @ x_h_k.transpose(-1, -2)) * self.scale).softmax(-1)
        out2 = (score2 @ x_h_v).transpose(1, 2).reshape(B * l, l, C)  # B*l,lw,C
        out2 = out2.reshape(B, l, l, C)  # B,h,w,C
        # =========================================
        out = out1 + out2  # B,h,w,2C

        x = out.reshape(B, N, C) + x
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, whethermlp=True):
        super().__init__()
        self.whethermlp = whethermlp
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Stem(nn.Module):
    def __init__(self, inc, outc):
        super(Stem, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc - inc, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(outc - inc),
            nn.ReLU(True),
            nn.Conv2d(outc - inc, outc - inc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outc - inc),
            nn.ReLU(True),
        )
        self.adp = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.adp(x)
        return torch.cat([x1, x2], 1)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=2, in_chans=64, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.projs = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projs(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def Token2image(x):
    b, n, c = x.size()
    l = int(math.sqrt(n))
    x = x.reshape(b, l, l, c).permute(0, 3, 1, 2)
    return x


class ln(nn.Module):
    def __init__(self, inc, outc):
        super(ln, self).__init__()
        self.ln1 = nn.Sequential(
            nn.Linear(inc, outc),
            nn.LayerNorm(outc),
            nn.GELU()
        )

    def forward(self, x):
        x = self.ln1(x)
        return x


class Branch(nn.Module):
    def __init__(self):
        super(Branch, self).__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

    def forward(self, x):
        x1 = self.branch(x)
        x2 = self.branch2(x1)
        x3 = self.branch3(x2)
        return x1, x2, x3


class ResBlock(nn.Module):
    def __init__(self, inc, outc, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outc)
        )
        if stride != 1 or inc != outc:
            self.short = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outc)
            )
        else:
            self.short = nn.Identity()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        s = self.short(x)
        return F.relu(x1 + s, True)


class CrossNet(nn.Module):
    def __init__(self, nclass, head):
        super(CrossNet, self).__init__()
        self.stem = Stem(inc=6, outc=64)
        self.stage1 = nn.Sequential(
            PatchEmbed(in_chans=64, embed_dim=128),
            Block(dim=128, num_heads=head[0]),
            Block(dim=128, num_heads=head[0]),
            Block(dim=128, num_heads=head[0]),
        )
        self.stage2 = nn.Sequential(
            PatchEmbed(in_chans=256, embed_dim=256),
            Block(dim=256, num_heads=head[1]),
            Block(dim=256, num_heads=head[1]),
            Block(dim=256, num_heads=head[1]),
            Block(dim=256, num_heads=head[1]),
        )
        self.stage3 = nn.Sequential(
            PatchEmbed(in_chans=512, embed_dim=512),
            Block(dim=512, num_heads=head[2]),
            Block(dim=512, num_heads=head[2]),
            Block(dim=512, num_heads=head[2]),
            Block(dim=512, num_heads=head[2]),
            Block(dim=512, num_heads=head[2]),
            Block(dim=512, num_heads=head[2]),
        )
        self.stage4 = nn.Sequential(
            PatchEmbed(in_chans=1024, embed_dim=1024),
            Block(dim=1024, num_heads=head[3]),
            Block(dim=1024, num_heads=head[3]),
            Block(dim=1024, num_heads=head[3]),
        )
        self.o1 = ln(256, 128)
        self.o2 = ln(512, 128)
        self.o3 = ln(1024, 128)

        self.outconv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(128, nclass, kernel_size=1)
        )

        # ============================================
        self.stem_cnn = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage1_cnn = nn.Sequential(
            ResBlock(64, 64, 1),
            ResBlock(64, 64, 1),
            ResBlock(64, 64, 1),
        )
        self.stage2_cnn = nn.Sequential(
            ResBlock(128, 128, 2),
            ResBlock(128, 128, 1),
            ResBlock(128, 128, 1),
            ResBlock(128, 128, 1),
        )
        self.stage3_cnn = nn.Sequential(
            ResBlock(256, 256, 2),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
            ResBlock(256, 256, 1),
        )
        self.stage4_cnn = nn.Sequential(
            ResBlock(512, 512, 2),
            ResBlock(512, 512, 1),
            ResBlock(512, 512, 1),
        )
        # ==============================
        self.c1x1_1 = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.c1x1_11 = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64)
        )
        # ================================
        self.c1x1_2 = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.c1x1_22 = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        # ====================================
        self.c1x1_3 = nn.Sequential(
            nn.Conv2d(256, 512, 1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.c1x1_33 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256)
        )
        # ====================================
        self.c1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.outc1 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.outc2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(32, nclass, kernel_size=1)
        )

    def forward(self, x, y):
    # def forward(self, imgs):
    #     x = imgs.clone()
    #     y = imgs.clone()
        x = torch.cat([x, y], 1)
        # =========================================
        b = self.stem_cnn(x)  # 1/4,64
        stem = self.stem(x)  # 1/2,64
        # =========================================
        b1 = self.stage1_cnn(b)  # 1/4,64
        stage1 = self.stage1(stem)  # 1/4,128
        # ==========================================
        stage2 = self.stage2(torch.cat([Token2image(stage1), self.c1x1_1(b1)], 1))  # 1/8,256
        b2 = self.stage2_cnn(torch.cat([b1, self.c1x1_11(Token2image(stage1))], 1))  # 1/8,128
        # ============================================
        stage3 = self.stage3(torch.cat([Token2image(stage2), self.c1x1_2(b2)], 1))  # 1/16,512
        b3 = self.stage3_cnn(torch.cat([b2, self.c1x1_22(Token2image(stage2))], 1))  # 1/16,256
        # ==========================================
        stage4 = self.stage4(torch.cat([Token2image(stage3), self.c1x1_3(b3)], 1))  # 1/32,1024
        b4 = self.stage4_cnn(torch.cat([b3, self.c1x1_33(Token2image(stage3))], 1))  # 1/32,512
        # ===================================================
        stage2_ = self.o1(stage2)  # b,32*32,128
        stage3_ = self.o2(stage3)  # b,16*16,128
        stage4_ = self.o3(stage4)  # b,8*8,128
        # =======================================
        # =======================================
        out1 = Token2image(stage2_)  # 1/8,128
        out2 = self.deconv1(Token2image(stage3_)) + F.upsample(Token2image(stage3_), scale_factor=2, mode='bilinear',
                                                               align_corners=True)  # 1/8,128
        out3 = self.deconv2(Token2image(stage4_)) + F.upsample(Token2image(stage4_), scale_factor=4, mode='bilinear',
                                                               align_corners=True)  # 1/8,128
        out = torch.cat([out1, out2, out3], 1)  # 1/8,384
        # =======================================
        out_cnn1 = b2
        out_cnn2 = self.c1(b3)
        out_cnn2 = self.deconv3(out_cnn2) + F.upsample(out_cnn2, scale_factor=2, mode='bilinear', align_corners=True)
        out_cnn3 = self.c2(b4)
        out_cnn3 = self.deconv4(out_cnn3) + F.upsample(out_cnn3, scale_factor=4, mode='bilinear', align_corners=True)
        out_cnn = torch.cat([out_cnn1, out_cnn2, out_cnn3], 1)  # 1/8,384
        # =========================================
        plus = self.outc1(out + out_cnn)  # 1/8,128
        sub = self.outc2(out - out_cnn)  # 1/8,128
        cat = torch.cat([plus, sub], 1)
        OUT = self.outconv(cat)
        # ===================================================
        return F.upsample(OUT, scale_factor=8, mode='bilinear', align_corners=True) + self.deconv5(cat)

if __name__ == "__main__":
    from thop import profile
    acabfnet = CrossNet(nclass=2, head=[4,8,16,32])#.cuda()

    input1 = torch.randn(1, 3, 256, 256)#.cuda()
    flops1, params1 = profile(acabfnet, inputs=(input1,))
    print("flops=G", flops1 / (1000 ** 3))
    print("parms=M", params1 / (1000 ** 2))
