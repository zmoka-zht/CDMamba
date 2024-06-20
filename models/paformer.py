import torch
import torch.nn as nn
from models.paformer_help.backbone_help import build_backbone
from models.paformer_help.transmodel import TransformerDecoder, Transformer
from einops import rearrange

class DUpsampling(nn.Module):
    def __init__(self, in_chan, n_class, scale=4, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, n_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        self.scale = scale

    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1)

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        x = x_permuted.permute(0, 3, 2, 1)

        return x

class token_encoder(nn.Module):
    def __init__(self, in_chan = 32, token_len = 8, heads = 8):
        super(token_encoder, self).__init__()
        self.token_len = token_len
        self.conv_a = nn.Conv2d(in_chan, token_len, kernel_size=1, padding=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len, in_chan))
        self.transformer = Transformer(dim=in_chan, depth=1, heads=heads, dim_head=64, mlp_dim=64, dropout=0)

    def forward(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()

        tokens = torch.einsum('bln, bcn->blc', spatial_attention, x)

        tokens += self.pos_embedding
        x = self.transformer(tokens)
        return x

class token_decoder(nn.Module):
    def __init__(self, in_chan = 32, heads = 8):
        super(token_decoder, self).__init__()
        self.transformer_decoder = TransformerDecoder(dim=in_chan, depth=1, heads=heads, dim_head=True, mlp_dim=in_chan*2, dropout=0,softmax=in_chan)

    def forward(self, x, m):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

class Classifier(nn.Module):
    def __init__(self, in_chan=32, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
                            nn.Conv2d(in_chan, in_chan//2, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.BatchNorm2d(in_chan//2),
                            nn.ReLU(),
                            nn.Conv2d(in_chan//2, n_class, kernel_size=3, padding=1, stride=1))
    def forward(self, x):
        x = self.head(x)
        return x

class Paformer(nn.Module):
    def __init__(self, n_class=2, backbone='resnet18', output_stride=16, img_chan=3, f_c = 64):
        super(Paformer, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.PriorFExtractor = build_backbone(backbone, output_stride, BatchNorm, img_chan, f_c)

        self.token_encoder = token_encoder(in_chan=f_c)
        self.token_decoder = token_decoder(in_chan=f_c*2)

        self.decoder = DUpsampling(in_chan=f_c*2, n_class = n_class)

    # def forward(self, imgs):
    #     img1 = imgs.clone()
    #     img2 = imgs.clone()
    def forward(self, img1, img2):

        body1, out1_s16 = self.PriorFExtractor(img1)
        body2, out2_s16 = self.PriorFExtractor(img2)

        x16 = torch.cat([out1_s16, out2_s16], dim=1)

        x16 = self.token_decoder(x16, torch.cat([self.token_encoder(body1), self.token_encoder(body2)], dim=2))

        out = self.decoder(x16)

        return out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

if __name__ == "__main__":
    paformer = Paformer()
    # img = torch.rand([8, 3, 256, 256])
    #
    # res = paformer(img, img)
    # print(res.shape)
    from thop import profile

    # caculate flops1
    input1 = torch.randn(1, 3, 256, 256)
    flops1, params1 = profile(paformer, inputs=(input1,))
    print("flops=G", flops1 / (1000 ** 3))
    print("parms=M", params1 / (1000 ** 2))