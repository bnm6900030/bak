
from natten import NeighborhoodAttention2D as NeighborhoodAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
from natten.functional import natten2dav, natten2dqkrpb
from torch.nn.functional import pad

from timm.models.layers import DropPath
from torch.nn.init import trunc_normal_


class NAttention(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim,
        channel,
        num_heads,
        kernel_size,
        dilation=1,
        bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size

        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = nn.Conv2d(channel, channel*3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channel*3, channel*3, kernel_size=3, stride=1, padding=1, groups=channel*3, bias=False)


        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.temperature = nn.Parameter(torch.ones(channel, 1, 1))


    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
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

        attn = attn * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten2dav(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=True,)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class EfficientMixAttnTransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            input_resolution,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            res_scale=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.mlp_ratio = mlp_ratio
        self.res_scale = res_scale

        self.attn = NAttention(
            input_resolution[0],
            channel=dim,
            kernel_size=7,
            dilation=2,
            num_heads=4,
            qk_scale=None,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ffn = FeedForward(dim, 2.66, False)

        self.norm2 = norm_layer(dim)

    def forward(self, x):
        # Mixed attention
        x = x + self.res_scale * self.drop_path(
            (self.attn(self.norm1(x)))
        )
        # FFN
        x = x + self.res_scale * self.drop_path(self.ffn(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution} "
            f"mlp_ratio={self.mlp_ratio}, res_scale={self.res_scale}"
        )

    def flops(self):
        pass
