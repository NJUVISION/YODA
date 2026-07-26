# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import math
class DepthConvBlock4(nn.Module):
    def __init__(self, in_ch, out_ch, slope_depth_conv=0.01, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConvFM(in_ch, out_ch, slope=slope_depth_conv, inplace=inplace),
            ConvFFN3(out_ch, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)
class ConvFFN3(nn.Module):
    def __init__(self, in_ch, inplace=False):
        super().__init__()
        expansion_factor = 2
        internal_ch = in_ch * expansion_factor
        self.conv = nn.Conv2d(in_ch, internal_ch * 2, 1)
        self.conv_out = nn.Conv2d(internal_ch, in_ch, 1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=inplace)

    def forward(self, x):
        identity = x
        x1, x2 = self.conv(x).chunk(2, 1)
        out = self.relu1(x1) + self.relu2(x2)
        return identity + self.conv_out(out)
class DepthConvFM(nn.Module):
    def __init__(self, in_ch, out_ch, slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)

        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity

class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x


class WSiLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = WSiLU()

    def forward(self, x):
        x1, x2 = self.silu(x).chunk(2, 1)
        return x1 + x2


class SubpelConv2x(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding=padding),
            nn.PixelShuffle(2),
        )
        self.padding = padding

        self.proxy = None

    def forward(self, x, to_cat=None, cat_at_front=True):
            return self.forward_torch(x, to_cat, cat_at_front)


    def forward_torch(self, x, to_cat=None, cat_at_front=True):
        out = self.conv(x)
        if to_cat is None:
            return out
        if cat_at_front:
            return torch.cat((to_cat, out), dim=1)
        return torch.cat((out, to_cat), dim=1)




class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=False, force_adaptor=False):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            WSiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4, 1),
            WSiLUChunkAdd(),
            nn.Conv2d(out_ch * 2, out_ch, 1),
        )
        self.apply(weights_init_xavier_normal)
        nn.init.constant_(self.ffn[2].weight, 0)
        if self.ffn[2].bias is not None:
            nn.init.constant_(self.ffn[2].bias, 0)

        self.proxy = None
        # 应用初始化函数


    def forward(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        return self.forward_torch(x, quant_step, to_cat, cat_at_front)


    def forward_torch(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc(x) + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        if quant_step is not None:
            out = out * quant_step
        if to_cat is not None:
            if cat_at_front:
                out = torch.cat((to_cat, out), dim=1)
            else:
                out = torch.cat((out, to_cat), dim=1)
        return out

def weights_init_xavier_normal(m):
    """Init the weights with kaiming uniform.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight, math.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

class ResidualBlockWithStride2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)
        self.apply(weights_init_xavier_normal)
    def forward(self, x):
        x = self.down(x)
        out = self.conv(x)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)
        self.apply(weights_init_xavier_normal)
    def forward(self, x):
        x = self.down(x)
        out = self.conv(x)
        return out


class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch, 1)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)
        self.apply(weights_init_xavier_normal)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out
