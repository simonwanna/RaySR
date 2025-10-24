"""
Modelling base class and utilities.
Credit: https://github.com/eugenesiow/super-image & https://github.com/luissen/ESRT
"""

import functools
import logging
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def make_layer(block: functools.partial, n_layers: int) -> nn.Sequential:
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class SCPA(nn.Module):
    """
    SCPA is modified from SCNet (Liu et al. (2020))
    Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf: int, reduction: int = 2, stride: int = 1, dilation: int = 1):
        super(SCPA, self).__init__()
        group_width = nf // reduction

        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)

        self.k1 = nn.Sequential(
            nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False
            )
        )

        self.PAConv = PixelAttentionConv(group_width)

        self.conv3 = nn.Conv2d(group_width * reduction, nf, kernel_size=1, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out_a = self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual

        return out


class PixelAttentionConv(nn.Module):
    def __init__(self, nf: int, k_size: int = 3):
        super(PixelAttentionConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out


class PixelAttentionBlock(nn.Module):
    def __init__(self, nf: int):
        super(PixelAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.PReLU(in_planes // ratio)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = avg_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv1(max_out)
        return self.sigmoid(x)


class BamBlock(nn.Module):
    def __init__(self, in_planes: int, reduction: int = 16):
        super(BamBlock, self).__init__()

        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca_ch = self.ca(x)
        sa_ch = self.sa(x)
        out = ca_ch.mul(sa_ch) * x
        return out


class CharbonnierLoss(nn.Module):
    """Similar to L1 but differentiable at 0."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps2))


class GradientLoss(nn.Module):
    """L1 loss on image gradients (Sobel)."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _grad(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        gx = F.conv2d(x, self.kx.repeat(C, 1, 1, 1), padding=1, groups=C)
        gy = F.conv2d(x, self.ky.repeat(C, 1, 1, 1), padding=1, groups=C)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._grad(x), self._grad(y))


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super(CALayer, self).__init__()
        self.avg_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.conv_du: nn.Sequential = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class one_conv(nn.Module):
    def __init__(self, inchanels: int, growth_rate: int, kernel_size: int = 3, relu: bool = True) -> None:
        super(one_conv, self).__init__()

        def wn(x: nn.Module) -> nn.Module:
            return torch.nn.utils.weight_norm(x)

        self.conv: nn.Conv2d = nn.Conv2d(
            inchanels, growth_rate, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1
        )
        self.flag: bool = relu
        self.conv1: nn.Conv2d = nn.Conv2d(
            growth_rate, inchanels, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1
        )
        if relu:
            self.relu: nn.PReLU = nn.PReLU(growth_rate)
        self.weight1: Scale = Scale(1)
        self.weight2: Scale = Scale(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.flag:
            output: torch.Tensor = self.weight1(x) + self.weight2(self.conv1(self.conv(x)))
        else:
            output = self.weight1(x) + self.weight2(self.conv1(self.relu(self.conv(x))))
        return output


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        relu: bool = True,
        bn: bool = False,
        bias: bool = False,
        up_size: int = 0,
        fan: bool = False,
    ) -> None:
        super(BasicConv, self).__init__()

        def wn(x: nn.Module) -> nn.Module:
            return torch.nn.utils.weight_norm(x)

        self.out_channels: int = out_planes
        self.in_channels: int = in_planes
        if fan:
            self.conv: nn.Module = nn.ConvTranspose2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        else:
            self.conv = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        self.bn: Optional[nn.BatchNorm2d] = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        )
        self.relu: Optional[nn.ReLU] = nn.ReLU(inplace=True) if relu else None
        self.up_size: int = up_size
        self.up_sample: Optional[nn.Upsample] = (
            nn.Upsample(size=(up_size, up_size), mode="bilinear") if up_size != 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x


class one_module(nn.Module):
    def __init__(self, n_feats: int):
        super(one_module, self).__init__()
        self.layer1 = one_conv(n_feats, n_feats // 2, 3)
        self.layer2 = one_conv(n_feats, n_feats // 2, 3)
        # self.layer3 = one_conv(n_feats, n_feats//2,3)
        self.layer4 = BasicConv(n_feats, n_feats, 3, 1, 1)
        self.alise = BasicConv(2 * n_feats, n_feats, 1, 1, 0)
        self.atten = CALayer(n_feats)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)
        self.weight3 = Scale(1)
        self.weight4 = Scale(1)
        self.weight5 = Scale(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # pdb.set_trace()
        x4 = self.layer4(self.atten(self.alise(torch.cat([self.weight2(x2), self.weight3(x1)], 1))))
        return self.weight4(x) + self.weight5(x4)


class Updownblock(nn.Module):
    def __init__(self, n_feats: int):
        super(Updownblock, self).__init__()
        self.encoder = one_module(n_feats)
        self.decoder_low = one_module(n_feats)  # nn.Sequential(one_module(n_feats),
        #                     one_module(n_feats),
        #                     one_module(n_feats))
        self.decoder_high = one_module(n_feats)
        self.alise = one_module(n_feats)
        self.alise2 = BasicConv(2 * n_feats, n_feats, 1, 1, 0)  # one_module(n_feats)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode="bilinear", align_corners=True)
        for i in range(5):
            x2 = self.decoder_low(x2)
        x3 = x2
        # x3 = self.decoder_low(x2)
        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode="bilinear", align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x


class Un(nn.Module):
    def __init__(self, n_feats: int, wn: Callable) -> None:
        super(Un, self).__init__()
        self.encoder1 = Updownblock(n_feats)
        self.encoder2 = Updownblock(n_feats)
        self.encoder3 = Updownblock(n_feats)
        self.reduce = default_conv(3 * n_feats, n_feats, 3)
        self.weight2 = Scale(1)
        self.weight1 = Scale(1)
        self.attention = MLABlock(n_feat=n_feats, dim=288)
        self.alise = default_conv(n_feats, n_feats, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out = self.encoder3(self.encoder2(self.encoder1(x)))
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        out = x3
        b, c, h, w = x3.shape
        out = self.attention(self.reduce(torch.cat([x1, x2, x3], dim=1)))
        out = out.permute(0, 2, 1)
        out = reverse_patches(out, (h, w), (3, 3), 1, 1)
        out = self.alise(out)

        return self.weight1(x) + self.weight2(out)


class Scale(nn.Module):
    def __init__(self, init_value: float = 1e-3) -> None:
        super().__init__()
        self.scale: nn.Parameter = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.scale


# class MeanShift(nn.Conv2d):
#     def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
#         super(MeanShift, self).__init__(3, 3, kernel_size=1)
#         std = torch.Tensor(rgb_std)
#         self.weight.data = torch.eye(3).view(3, 3, 1, 1)
#         self.weight.data.div_(std.view(3, 1, 1, 1))
#         self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
#         self.bias.data.div_(std)
#         self.requires_grad = False


def reverse_patches(
    images: torch.Tensor, out_size: Tuple[int, int], ksizes: Tuple[int, int], strides: int, padding: int
) -> torch.Tensor:
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    unfold: torch.nn.Fold = torch.nn.Fold(
        output_size=out_size, kernel_size=ksizes, dilation=1, padding=padding, stride=strides
    )
    patches: torch.Tensor = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


class Upsampler(nn.Sequential):
    def __init__(
        self,
        conv: Callable,
        scale: int,
        n_feats: int,
        bn: bool = False,
        act: Union[bool, str] = False,
        bias: bool = True,
    ) -> None:
        m: List[nn.Module] = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type = nn.ReLU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features // 4
        self.fc1: nn.Linear = nn.Linear(in_features, hidden_features)
        self.act: nn.Module = act_layer()
        self.fc2: nn.Linear = nn.Linear(hidden_features, out_features)
        self.drop: nn.Dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MLABlock(nn.Module):
    def __init__(
        self,
        n_feat: int = 64,
        dim: int = 768,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.ReLU,
        norm_layer: type = nn.LayerNorm,
    ) -> None:
        super(MLABlock, self).__init__()
        self.dim: int = dim
        self.atten: EffAttention = EffAttention(
            self.dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0
        )
        self.norm1: nn.LayerNorm = nn.LayerNorm(self.dim)
        self.mlp: Mlp = Mlp(in_features=dim, hidden_features=dim // 4, act_layer=act_layer, drop=drop)
        self.norm2: nn.LayerNorm = nn.LayerNorm(self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B: int = x.shape[0]
        x = extract_image_patches(x, ksizes=[3, 3], strides=[1, 1], rates=[1, 1], padding="same")
        x = x.permute(0, 2, 1)
        x = x + self.atten(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def same_padding(images: torch.Tensor, ksizes: List[int], strides: List[int], rates: List[int]) -> torch.Tensor:
    assert len(images.size()) == 4
    batch_size: int
    channel: int
    rows: int
    cols: int
    batch_size, channel, rows, cols = images.size()
    out_rows: int = (rows + strides[0] - 1) // strides[0]
    out_cols: int = (cols + strides[1] - 1) // strides[1]
    effective_k_row: int = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col: int = (ksizes[1] - 1) * rates[1] + 1
    padding_rows: int = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols: int = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top: int = int(padding_rows / 2.0)
    padding_left: int = int(padding_cols / 2.0)
    padding_bottom: int = padding_rows - padding_top
    padding_right: int = padding_cols - padding_left
    paddings: Tuple[int, int, int, int] = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(
    images: torch.Tensor, ksizes: List[int], strides: List[int], rates: List[int], padding: str = "same"
) -> torch.Tensor:
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ["same", "valid"]
    batch_size: int
    channel: int
    height: int
    width: int
    batch_size, channel, height, width = images.size()

    if padding == "same":
        images = same_padding(images, ksizes, strides, rates)
    elif padding == "valid":
        pass
    else:
        raise NotImplementedError(
            'Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding)
        )

    unfold: torch.nn.Unfold = torch.nn.Unfold(kernel_size=ksizes, dilation=rates, padding=0, stride=strides)
    patches: torch.Tensor = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


class EffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        head_dim: int = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale: float = qk_scale or head_dim**-0.5

        self.reduce: nn.Linear = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.qkv: nn.Linear = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.proj: nn.Linear = nn.Linear(dim // 2, dim)
        self.attn_drop: nn.Dropout = nn.Dropout(attn_drop)
        print("scale", self.scale)
        print(dim)
        print(dim // num_heads)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        B, N, C = x.shape
        # pdb.set_trace()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # v = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv: 3*16*8*37*96
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # pdb.set_trace()

        q_all = torch.split(q, math.ceil(N // 4), dim=-2)
        k_all = torch.split(k, math.ceil(N // 4), dim=-2)
        v_all = torch.split(v, math.ceil(N // 4), dim=-2)

        output = []
        for q, k, v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1)) * self.scale  # 16*8*37*37
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            trans_x = (attn @ v).transpose(1, 2)  # .reshape(B, N, C)

            output.append(trans_x)
        # pdb.set_trace()
        # attn = torch.cat(att, dim=-2)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C) #16*37*768
        x = torch.cat(output, dim=1)
        x = x.reshape(B, N, C)
        # pdb.set_trace()
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


def default_conv(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = True, groups: int = 1
) -> nn.Conv2d:
    def wn(x: nn.Module) -> nn.Module:
        return torch.nn.utils.weight_norm(x)

    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, groups=groups)
