import torch
import torch.nn as nn

class SeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bias=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = kernel_size

        self.depthwise = nn.Conv1d(in_channels=self.in_ch,
                                   out_channels=self.in_ch,
                                   kernel_size=self.k,
                                   groups=self.in_ch,
                                   padding=self.k // 2,
                                   bias=False)
        self.pointwise = nn.Conv1d(in_channels=self.in_ch,
                                   out_channels=self.out_ch,
                                   kernel_size=1,
                                   padding=0,
                                   bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_ch))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_normal_(self.depthwise.weight)
        nn.init.kaiming_normal_(self.pointwise.weight)

    def forward(self, x):
        out = self.pointwise(self.depthwise(x.transpose(1, 2))).transpose(1, 2)
        if self.bias is not None:
            out += self.bias
        return out

if __name__ == "__main__":
    spconv = SeparableConv1d(in_ch=1024, out_ch=512, bias=False, kernel_size=3)
    img = torch.randn([4, 256, 1024])

    res = spconv(img)
    print(res.shape)