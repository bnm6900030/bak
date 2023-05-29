from fairscale.nn import checkpoint_wrapper
import torch.nn as nn

from natten import NeighborhoodAttention2D as NeighborhoodAttention
import torch.nn.functional as F
from natten.functional import natten2dqkrpb, natten2dav
from timm.models.layers import trunc_normal_, DropPath
import torch

from basicsr.archs.utils import LayerNorm
from basicsr.utils.registry import ARCH_REGISTRY


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=True, )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # bhwc 2 bchw
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = x.permute(0, 2, 3, 1)  # bchw 2 bhwc
        return x


class SGForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super(SGForward, self).__init__()

        ffn_channel = dim * ffn_expansion_factor
        self.conv4 = nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=dim, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # bhwc 2 bchw

        x = self.conv4(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.conv5(x) * self.beta

        x = x.permute(0, 2, 3, 1)  # bchw 2 bhwc
        return x


class NAttention(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
            self,
            dim,
            num_heads,
            kernel_size,
            dilation=1,
            bias=True,
            qk_scale=None,
            attn_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads // 2
        self.scale = qk_scale or self.head_dim ** -0.5
        assert (
                kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
                dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1

        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        qkv = (x.reshape(B, H, W, 3, self.num_heads, self.head_dim)
               .permute(3, 0, 4, 1, 2, 5)
               )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten2dav(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C // 3)
        return x

    def extra_repr(self) -> str:
        return (
                f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
                + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
                + f"rel_pos_bias={self.rpb is not None}"
        )


class Attention(nn.Module):

    def __init__(
            self, dim, num_heads=8, attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2
        # self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        qkv = x.reshape(B, H*W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, H, W, C // 3)

        return x


class MixedAttention(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size,
            dilation,
            qkv_bias,
            num_heads,
            qk_scale,
            attn_drop,
            proj_drop,
    ):
        super(MixedAttention, self).__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.dim = dim
        self.na_attn = NAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
        )
        self.self_attn = Attention(dim, num_heads, attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        # qkv projection
        qkv = self.qkv(x)
        qkv_na, qkv_self = torch.split(qkv, C * 3 // 2, dim=-1)

        # attention
        x_na = self.na_attn(qkv_na)
        x_self = self.self_attn(qkv_self)
        x = torch.cat([x_na, x_self], dim=-1)

        # output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NATransformerLayer(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            kernel_size=7,
            dilation=1,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # self.attn = MixedAttention(
        #     dim,
        #     kernel_size=kernel_size,
        #     dilation=dilation,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        # )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=1e-6)
        # self.mlp = FeedForward(dim, 2.66, False)
        self.mlp = SGForward(dim, 2)

    def forward(self, x):
        shortcut = x
        x = shortcut + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.prj = nn.Conv2d(dim, dim, 3, 1, 1)
        self.downsample_layer = nn.Sequential(
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # b c h w
        x = self.downsample_layer(x)
        x = x.permute(0, 2, 3, 1)  # b h w c
        return x


class BasicLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            num_heads,
            kernel_size,
            dilations=None,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=None,
            norm_layer=nn.LayerNorm,
            downsample=None,
            is_conv=False,
    ):

        super().__init__()
        if drop_path is None:
            drop_path = []
        self.dim = dim
        self.depth = depth

        self.is_conv = is_conv
        if is_conv:
            self.blocks = nn.ModuleList(
                [nn.Sequential(
                    *[Block(dim=dim, drop_path=drop_path[i]) for i in range(depth)]
                )]
            )
        else:
            # build blocks
            self.blocks = nn.ModuleList(
                [
                    checkpoint_wrapper(NATransformerLayer(
                        dim=dim,
                        num_heads=num_heads,
                        kernel_size=kernel_size,
                        dilation=1 if dilations is None else dilations[i],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                    ))
                    for i in range(depth)
                ]
            )
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        if self.is_conv:
            x = x.permute(0, 3, 1, 2)  # b c h w
        for blk in self.blocks:
            x = blk(x)
        if self.is_conv:
            x = x.permute(0, 2, 3, 1)  # b h w c

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.norm = None if norm_layer is None else norm_layer(embed_dim, eps=1e-6)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # b h w c
        if self.norm is not None:
            x = self.norm(x)
        return x


@ARCH_REGISTRY.register()
class MYIR4(nn.Module):
    def __init__(
            self,
            in_chans=6,
            embed_dim=48,
            depths=None,
            num_heads=None,
            is_conv_list=None,
            kernel_size=7,
            dilations=None,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=False,
            is_checkpoint=True,
            **kwargs
    ):
        super().__init__()

        if depths is None:
            depths = [3, 3, 9, 3, ]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if is_conv_list is None:
            is_conv_list = [False, False, False, False]
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        # self.patch_embed = checkpoint_wrapper(self.patch_embed)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                # dim=int(embed_dim * 2 ** i_layer),
                dim=embed_dim,
                depth=depths[i_layer],
                is_conv=is_conv_list[i_layer],
                num_heads=num_heads[i_layer],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging,
            )
            # if i_layer % 2 == 0:
            # layer = checkpoint_wrapper(layer)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, 3, 3, 1, 1, 1)

        # self.conv_last = checkpoint_wrapper(self.conv_last)
        # self.conv_after_body = checkpoint_wrapper(self.conv_after_body)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"rpb"}

    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # b c h w
        return x

    def forward(self, x, C):
        # single = C
        x = self.patch_embed(x)
        f_x = self.forward_features(x)
        res = self.conv_after_body(f_x) + x.permute(0, 3, 1, 2)
        x = self.conv_last(res) + C
        return x, None


if __name__ == '__main__':
    # model = MYIR(num_heads=[3, 6, 12, 24],
    #              embed_dim=96,
    #              depths=[2, 2, 6, 2],
    #              is_conv_list=[False, False, False, False],
    #              dilations=[
    #                  [1, 8],
    #                  [1, 4],
    #                  [1, 2, 1, 2, 1, 2],
    #                  [1, 1],
    #              ]
    #              )
    model = MYIR4(num_heads=[2, 6, 12, 24],
                  embed_dim=48,
                  depths=[2, 2, 6, 2],
                  is_conv_list=[False, False, False, False, ],
                  )
    model.cuda()
    from torchsummary import summary
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    summary(model, [(6, 256, 256), (3, 256, 256)])
