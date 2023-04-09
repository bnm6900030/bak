import math
from abc import ABC
from math import prod
from natten import NeighborhoodAttention2D as NeighborhoodAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
from natten.functional import natten2dav, natten2dqkrpb
from torch.nn.functional import pad
from common.mixed_attn_block import (
    AnchorProjection,
    CAB,
    CPB_MLP,
    QKVProjection,
)
from common.ops import (
    window_partition,
    window_reverse, bchw_to_blc, blc_to_bchw, blc_to_bhwc,
)

from timm.models.layers import DropPath
from torch.nn.init import trunc_normal_


class AffineTransform(nn.Module):
    r"""Affine transformation of the attention map.
    The window could be a square window or a stripe window. Supports attention between different window sizes
    """

    def __init__(self, num_heads):
        super(AffineTransform, self).__init__()
        logit_scale = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(logit_scale, requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = CPB_MLP(2, num_heads)

    def forward(self, attn, relative_coords_table, relative_position_index, mask):
        B_, H, N1, N2 = attn.shape
        # logit scale
        attn = attn * torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()

        bias_table = self.cpb_mlp(relative_coords_table)  # 2*Wh-1, 2*Ww-1, num_heads
        bias_table = bias_table.view(-1, H)

        bias = bias_table[relative_position_index.view(-1)]
        bias = bias.view(N1, N2, -1).permute(2, 0, 1).contiguous()
        # nH, Wh*Ww, Wh*Ww
        bias = 16 * torch.sigmoid(bias)
        attn = attn + bias.unsqueeze(0)

        # W-MSA/SW-MSA
        # shift attention mask
        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, H, N1, N2) + mask
            attn = attn.view(-1, H, N1, N2)

        return attn


def _get_stripe_info(stripe_size_in, stripe_groups_in, stripe_shift, input_resolution):
    stripe_size, shift_size = [], []
    for s, g, d in zip(stripe_size_in, stripe_groups_in, input_resolution):
        if g is None:
            stripe_size.append(s)
            shift_size.append(s // 2 if stripe_shift else 0)
        else:
            stripe_size.append(d // g)
            shift_size.append(0 if g == 1 else d // (g * 2))
    return stripe_size, shift_size


class Attention(ABC, nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def attn(self, q, k, v, attn_transform, table, index, mask, reshape=True):
        # q, k, v: # nW*B, H, wh*ww, dim
        # cosine attention map
        B_, _, H, head_dim = q.shape
        if self.euclidean_dist:
            # print("use euclidean distance")
            attn = torch.norm(q.unsqueeze(-2) - k.unsqueeze(-3), dim=-1)
        else:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn_transform(attn, table, index, mask)
        # attention
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v  # B_, H, N1, head_dim
        if reshape:
            x = x.transpose(1, 2).reshape(B_, -1, H * head_dim)
        # B_, N, C
        return x


class WindowAttention(Attention):
    r"""Window attention. QKV is the input to the forward method.
    Args:
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
            self,
            input_resolution,
            window_size,
            num_heads,
            window_shift=False,
            attn_drop=0.0,
            pretrained_window_size=[0, 0],
            args=None,
    ):

        super(WindowAttention, self).__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.shift_size = window_size[0] // 2 if window_shift else 0
        self.euclidean_dist = args.euclidean_dist

        self.attn_transform = AffineTransform(num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, x_size, table, index, mask):
        """
        Args:
            qkv: input QKV features with shape of (B, L, 3C)
            x_size: use x_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        B, L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            qkv = torch.roll(
                qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )

        # partition windows
        qkv = window_partition(qkv, self.window_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(self.window_size), C)  # nW*B, wh*ww, C

        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) #3, B,HEAD-1
        q, k, v = qkv[0], qkv[1], qkv[2]  # nW*B, H, wh*ww, dim

        # attention
        x = self.attn(q, k, v, self.attn_transform, table, index, mask)

        # merge windows
        x = x.view(-1, *self.window_size, C // 3)
        x = window_reverse(x, self.window_size, x_size)  # B, H, W, C/3

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, L, C // 3)

        return x

    def extra_repr(self) -> str:
        return (
            f"window_size={self.window_size}, shift_size={self.shift_size}, "
            f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
        )

    def flops(self, N):
        pass

class NAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            kernel_size,
            dilation=1,
            bias=True,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
    ):

        super(NAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert (
                kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
                dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."

        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(int(dim * 1.5), int(dim *3), bias=qkv_bias)
        self.after_qkv = nn.Linear(int(dim), int(dim / 2), bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, x_size):
        Hp, Wp = x_size
        H, W = int(Hp), int(Wp)
        B, L, C = x.shape
        x = x.view(B, Hp, Wp, C)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, H, W, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten2dav(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, int(C / 1.5))
        x = self.after_qkv(x)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return x.view(B, L, C // 3)



class AnchorStripeAttention(Attention):
    r"""Stripe attention
    Args:
        stripe_size (tuple[int]): The height and width of the stripe.
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_stripe_size (tuple[int]): The height and width of the stripe in pre-training.
    """

    def __init__(
            self,
            input_resolution,
            stripe_size,
            stripe_groups,
            stripe_shift,
            num_heads,
            attn_drop=0.0,
            pretrained_stripe_size=[0, 0],
            anchor_window_down_factor=1,
            args=None,
    ):

        super(AnchorStripeAttention, self).__init__()
        self.input_resolution = input_resolution
        self.stripe_size = stripe_size  # Wh, Ww
        self.stripe_groups = stripe_groups
        self.stripe_shift = stripe_shift
        self.num_heads = num_heads
        self.pretrained_stripe_size = pretrained_stripe_size
        self.anchor_window_down_factor = anchor_window_down_factor
        self.euclidean_dist = args.euclidean_dist

        self.attn_transform1 = AffineTransform(num_heads)
        self.attn_transform2 = AffineTransform(num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
            self, qkv, anchor, x_size, table, index_a2w, index_w2a, mask_a2w, mask_w2a
    ):
        """
        Args:
            qkv: input features with shape of (B, L, C)
            anchor:
            x_size: use stripe_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        B, L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)

        stripe_size, shift_size = _get_stripe_info(
            self.stripe_size, self.stripe_groups, self.stripe_shift, x_size
        )
        anchor_stripe_size = [s // self.anchor_window_down_factor for s in stripe_size]
        anchor_shift_size = [s // self.anchor_window_down_factor for s in shift_size]
        # cyclic shift
        if self.stripe_shift:
            qkv = torch.roll(qkv, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            anchor = torch.roll(
                anchor,
                shifts=(-anchor_shift_size[0], -anchor_shift_size[1]),
                dims=(1, 2),
            )

        # partition windows
        qkv = window_partition(qkv, stripe_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(stripe_size), C)  # nW*B, wh*ww, C
        anchor = window_partition(anchor, anchor_stripe_size)
        anchor = anchor.view(-1, prod(anchor_stripe_size), C // 3)

        B_, N1, _ = qkv.shape
        N2 = anchor.shape[1]
        qkv = qkv.reshape(B_, N1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        anchor = anchor.reshape(B_, N2, self.num_heads, -1).permute(0, 2, 1, 3)

        # attention
        x = self.attn(
            anchor, k, v, self.attn_transform1, table, index_a2w, mask_a2w, False
        )
        x = self.attn(q, anchor, x, self.attn_transform2, table, index_w2a, mask_w2a)

        # merge windows
        x = x.view(B_, *stripe_size, C // 3)
        x = window_reverse(x, stripe_size, x_size)  # B H' W' C

        # reverse the shift
        if self.stripe_shift:
            x = torch.roll(x, shifts=shift_size, dims=(1, 2))

        x = x.view(B, H * W, C // 3)
        return x

    def extra_repr(self) -> str:
        return (
            f"stripe_size={self.stripe_size}, stripe_groups={self.stripe_groups}, stripe_shift={self.stripe_shift}, "
            f"pretrained_stripe_size={self.pretrained_stripe_size}, num_heads={self.num_heads}, anchor_window_down_factor={self.anchor_window_down_factor}"
        )

    def flops(self, N):
        pass


class MixedAttention(nn.Module):
    r"""Mixed window attention and stripe attention
    Args:
        dim (int): Number of input channels.
        stripe_size (tuple[int]): The height and width of the stripe.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_stripe_size (tuple[int]): The height and width of the stripe in pre-training.
    """

    def __init__(
            self,
            dim,
            input_resolution,
            num_heads_w,
            num_heads_s,
            window_size,
            window_shift,
            stripe_size,
            stripe_groups,
            stripe_shift,
            qkv_bias=True,
            qkv_proj_type="linear",
            anchor_proj_type="separable_conv",
            anchor_one_stage=True,
            anchor_window_down_factor=1,
            attn_drop=0.0,
            proj_drop=0.0,
            pretrained_window_size=[0, 0],
            pretrained_stripe_size=[0, 0],
            args=None,
    ):

        super(MixedAttention, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.args = args
        # print(args)
        self.qkv = QKVProjection(dim, qkv_bias, qkv_proj_type, args)
        # anchor is only used for stripe attention
        self.anchor = AnchorProjection(
            dim, anchor_proj_type, anchor_one_stage, anchor_window_down_factor, args
        )

        # self.window_attn = WindowAttention(
        #     input_resolution,
        #     window_size,
        #     num_heads_w,
        #     window_shift,
        #     attn_drop,
        #     pretrained_window_size,
        #     args,
        # )
        self.NA_attn = NAttention(
            dim,
            kernel_size=7,
            dilation=2,
            num_heads=num_heads_s,
            qkv_bias=qkv_bias,
            qk_scale=None,
            attn_drop=attn_drop,
        )
        self.stripe_attn = AnchorStripeAttention(
            input_resolution,
            stripe_size,
            stripe_groups,
            stripe_shift,
            num_heads_s,
            attn_drop,
            pretrained_stripe_size,
            anchor_window_down_factor,
            args,
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_size, table_index_mask):
        """
        Args:
            x: input features with shape of (B, L, C)
            stripe_size: use stripe_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        B, L, C = x.shape

        # qkv projection
        qkv = self.qkv(x, x_size)
        qkv_window, qkv_stripe = torch.split(qkv, C * 3 // 2, dim=-1)
        # anchor projection
        anchor = self.anchor(x, x_size)

        # attention
        # x_window = self.window_attn(
        #     qkv_window, x_size, *self._get_table_index_mask(table_index_mask, True)
        # )
        x_window = self.NA_attn(qkv_window, x_size)
        x_stripe = self.stripe_attn(
            qkv_stripe,
            anchor,
            x_size,
            *self._get_table_index_mask(table_index_mask, False),
        )
        x = torch.cat([x_window, x_stripe], dim=-1)

        # output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _get_table_index_mask(self, table_index_mask, window_attn=True):
        if window_attn:
            return (
                table_index_mask["table_w"],
                table_index_mask["index_w"],
                table_index_mask["mask_w"],
            )
        else:
            return (
                table_index_mask["table_s"],
                table_index_mask["index_a2w"],
                table_index_mask["index_w2a"],
                table_index_mask["mask_a2w"],
                table_index_mask["mask_w2a"],
            )

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}"

    def flops(self, N):
        pass


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=True,)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, x_size):
        x = blc_to_bchw(x, x_size).contiguous()
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        return bchw_to_blc(x)


class EfficientMixAttnTransformerBlock(nn.Module):
    r"""Mix attention transformer block with shared QKV projection and output projection for mixed attention modules.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_stripe_size (int): Window size in pre-training.
        attn_type (str, optional): Attention type. Default: cwhv.
                    c: residual blocks
                    w: window attention
                    h: horizontal stripe attention
                    v: vertical stripe attention
    """

    def __init__(
            self,
            dim,
            input_resolution,
            num_heads_w,
            num_heads_s,
            window_size=7,
            window_shift=False,
            stripe_size=[8, 8],
            stripe_groups=[None, None],
            stripe_shift=False,
            stripe_type="H",
            mlp_ratio=4.0,
            qkv_bias=True,
            qkv_proj_type="linear",
            anchor_proj_type="separable_conv",
            anchor_one_stage=True,
            anchor_window_down_factor=1,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            pretrained_window_size=[0, 0],
            pretrained_stripe_size=[0, 0],
            res_scale=1.0,
            args=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads_w = num_heads_w
        self.num_heads_s = num_heads_s
        self.window_size = window_size
        self.window_shift = window_shift
        self.stripe_shift = stripe_shift
        self.stripe_type = stripe_type
        self.args = args
        if self.stripe_type == "W":
            self.stripe_size = stripe_size[::-1]
            self.stripe_groups = stripe_groups[::-1]
        else:
            self.stripe_size = stripe_size
            self.stripe_groups = stripe_groups
        self.mlp_ratio = mlp_ratio
        self.res_scale = res_scale

        self.attn1 = MixedAttention(
            dim,
            input_resolution,
            num_heads_w,
            num_heads_s,
            window_size,
            window_shift,
            self.stripe_size,
            self.stripe_groups,
            stripe_shift,
            qkv_bias,
            qkv_proj_type,
            anchor_proj_type,
            anchor_one_stage,
            anchor_window_down_factor,
            attn_drop,
            drop,
            pretrained_window_size,
            pretrained_stripe_size,
            args,
        )
        # self.attn = NeighborhoodAttention(
        #     dim,
        #     kernel_size=7,
        #     dilation=2,
        #     num_heads=4,
        #     qkv_bias=qkv_bias,
        #     qk_scale=None,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        # )
        self.norm1 = norm_layer(dim)
        if self.args.local_connection:
            self.conv = CAB(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ffn = FeedForward(dim, 2.66, False)

        self.norm2 = norm_layer(dim)

    def _get_table_index_mask(self, all_table_index_mask):
        table_index_mask = {
            "table_w": all_table_index_mask["table_w"],
            "index_w": all_table_index_mask["index_w"],
        }
        if self.stripe_type == "W":
            table_index_mask["table_s"] = all_table_index_mask["table_sv"]
            table_index_mask["index_a2w"] = all_table_index_mask["index_sv_a2w"]
            table_index_mask["index_w2a"] = all_table_index_mask["index_sv_w2a"]
        else:
            table_index_mask["table_s"] = all_table_index_mask["table_sh"]
            table_index_mask["index_a2w"] = all_table_index_mask["index_sh_a2w"]
            table_index_mask["index_w2a"] = all_table_index_mask["index_sh_w2a"]
        if self.window_shift:
            table_index_mask["mask_w"] = all_table_index_mask["mask_w"]
        else:
            table_index_mask["mask_w"] = None
        if self.stripe_shift:
            if self.stripe_type == "W":
                table_index_mask["mask_a2w"] = all_table_index_mask["mask_sv_a2w"]
                table_index_mask["mask_w2a"] = all_table_index_mask["mask_sv_w2a"]
            else:
                table_index_mask["mask_a2w"] = all_table_index_mask["mask_sh_a2w"]
                table_index_mask["mask_w2a"] = all_table_index_mask["mask_sh_w2a"]
        else:
            table_index_mask["mask_a2w"] = None
            table_index_mask["mask_w2a"] = None
        return table_index_mask

    def forward(self, x, x_size, all_table_index_mask):
        # Mixed attention

        table_index_mask = self._get_table_index_mask(all_table_index_mask)
        # if self.args.local_connection:
        #     x = self.norm1(x)
        #     x=blc_to_bhwc(x,x_size)
        #     x = (
        #         x
        #         # + self.res_scale * self.drop_path(self.attn(self.norm1(x), x_size, table_index_mask))
        #         + self.res_scale * self.drop_path(self.attn(x))
        #         # + self.conv(x, x_size)
        #     )
        #     x=x.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
        # else:
        x = x + self.res_scale * self.drop_path(
            (self.attn1(self.norm1(x), x_size, table_index_mask))
        )
        # FFN
        x = x + self.res_scale * self.drop_path(self.ffn(self.norm2(x), x_size))
        x = blc_to_bchw(x, x_size).contiguous()
        return bchw_to_blc(x)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads=({self.num_heads_w}, {self.num_heads_s}), "
            f"window_size={self.window_size}, window_shift={self.window_shift}, "
            f"stripe_size={self.stripe_size}, stripe_groups={self.stripe_groups}, stripe_shift={self.stripe_shift}, self.stripe_type={self.stripe_type}, "
            f"mlp_ratio={self.mlp_ratio}, res_scale={self.res_scale}"
        )

    def flops(self):
        pass
