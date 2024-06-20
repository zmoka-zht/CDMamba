import torch
import torch.nn as nn
import torch.nn.functional as F

# class TwoLayerConv2d(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
#                             padding=kernel_size // 2, stride=1, bias=False),
#                          nn.BatchNorm2d(in_channels),
#                          nn.ReLU(),
#                          nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
#                             padding=kernel_size // 2, stride=1)
#                          )

class TwoLayerConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(TwoLayerConv2d, self).__init__()

        self.Conv2d_1 = nn.Conv2d(in_channel, in_channel, kernel_size,
                                padding=kernel_size //2, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()
        self.Conv2d_2 = nn.Conv2d(in_channel, out_channel, kernel_size,
                                  padding=kernel_size //2, stride=1)

    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.Conv2d_2(x)

        return x


