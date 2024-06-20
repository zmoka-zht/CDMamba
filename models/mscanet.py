import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mscanet_help.backbone_help import build_backbone
from models.mscanet_help.transmodel import TransformerDecoder, Transformer
from einops import rearrange


class token_encoder(nn.Module):
    def __init__(self, in_chan = 32, token_len = 4, heads = 8):
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
    def __init__(self, in_chan = 32, size = 32, heads = 8):
        super(token_decoder, self).__init__()
        self.pos_embedding_decoder = nn.Parameter(torch.randn(1, in_chan, size, size))
        self.transformer_decoder = TransformerDecoder(dim=in_chan, depth=1, heads=heads, dim_head=True, mlp_dim=in_chan*2, dropout=0,softmax=in_chan)

    def forward(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x


class context_aggregator(nn.Module):
    def __init__(self, in_chan=32, size=32):
        super(context_aggregator, self).__init__()
        self.token_encoder = token_encoder(in_chan=in_chan, token_len=4)
        self.token_decoder = token_decoder(in_chan = 32, size = size, heads = 8)

    def forward(self, feature):
        token = self.token_encoder(feature)
        out = self.token_decoder(feature, token)
        return out

class Classifier(nn.Module):
    def __init__(self, in_chan=32, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
                            nn.Conv2d(in_chan * 2, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.BatchNorm2d(in_chan),
                            nn.ReLU(),
                            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1))
    def forward(self, x):
        x = self.head(x)
        return x

class MSCACDNet(nn.Module):
    def __init__(self,  backbone='resnet18', output_stride=16, img_size = 256, img_chan=3, chan_num = 32, n_class =2):
        super(MSCACDNet, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, img_chan)

        self.CA_s16 = context_aggregator(in_chan=chan_num, size=img_size//16)
        self.CA_s8 = context_aggregator(in_chan=chan_num, size=img_size//8)
        self.CA_s4 = context_aggregator(in_chan=chan_num, size=img_size//4)

        self.conv_s8 = nn.Conv2d(chan_num*2, chan_num, kernel_size=3, padding=1)
        self.conv_s4 = nn.Conv2d(chan_num*2, chan_num, kernel_size=3, padding=1)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)

        self.classifier1 = Classifier(n_class = n_class)
        self.classifier2 = Classifier(n_class = n_class)
        self.classifier3 = Classifier(n_class = n_class)

    # def forward(self, imgs):
    #     img1 = imgs.clone()
    #     img2 = imgs.clone()
    def forward(self, img1, img2):
        # CNN backbone, feature extractor
        out1_s16, out1_s8, out1_s4 = self.backbone(img1)
        out2_s16, out2_s8, out2_s4 = self.backbone(img2)

        # context aggregate (scale 16, scale 8, scale 4)
        x1_s16= self.CA_s16(out1_s16)
        x2_s16 = self.CA_s16(out2_s16)

        x16 = torch.cat([x1_s16, x2_s16], dim=1)
        x16 = F.interpolate(x16, size=img1.shape[2:], mode='bicubic', align_corners=True)
        x16 = self.classifier1(x16)

        out1_s8 = self.conv_s8(torch.cat([self.upsamplex2(x1_s16), out1_s8], dim=1))
        out2_s8 = self.conv_s8(torch.cat([self.upsamplex2(x2_s16), out2_s8], dim=1))

        x1_s8 = self.CA_s8(out1_s8)
        x2_s8 = self.CA_s8(out2_s8)

        x8 = torch.cat([x1_s8, x2_s8], dim=1)
        x8 = F.interpolate(x8, size=img1.shape[2:], mode='bicubic', align_corners=True)
        x8 = self.classifier2(x8)

        out1_s4 = self.conv_s4(torch.cat([self.upsamplex2(x1_s8), out1_s4], dim=1))
        out2_s4 = self.conv_s4(torch.cat([self.upsamplex2(x2_s8), out2_s4], dim=1))

        x1 = self.CA_s4(out1_s4)
        x2 = self.CA_s4(out2_s4)

        x = torch.cat([x1, x2], dim=1)
        x = F.interpolate(x, size=img1.shape[2:], mode='bicubic', align_corners=True)
        x = self.classifier3(x)

        return x, x8, x16

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

if __name__ == "__main__":
    mscanet = MSCACDNet()
    # img = torch.rand([8, 3, 512, 512])
    #
    # res = mscanet(img, img)
    # print(res[0].shape)
    # print(res[1].shape)
    # print(res[2].shape)
    from thop import profile

    # caculate flops1
    input1 = torch.randn(1, 3, 256, 256)
    flops1, params1 = profile(mscanet, inputs=(input1,))
    print("flops=G", flops1 / (1000 ** 3))
    print("parms=M", params1 / (1000 ** 2))