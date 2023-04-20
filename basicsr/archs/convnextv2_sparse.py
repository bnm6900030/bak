import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.utils.checkpoint import checkpoint
from basicsr.archs.utils import (
    MinkowskiLayerNorm,
    MinkowskiGRN,
    MinkowskiDropPath,
)
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU
)
from MinkowskiOps import (
    to_sparse,
)


class PatchMerging(nn.Module):
    def __init__(self, dim, D):
        super().__init__()
        self.downsample_layer = nn.Sequential(
            MinkowskiLayerNorm(dim, eps=1e-6, ),
            MinkowskiConvolution(dim, dim, kernel_size=3, stride=1, bias=True,dimension=D),
        )

    def forward(self, x):
        x = self.downsample_layer(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.norm = None if norm_layer is None else norm_layer(embed_dim, eps=1e-6)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Block(nn.Module):
    """ Sparse ConvNeXtV2 Block.
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., D=3):
        super().__init__()
        self.dwconv = MinkowskiDepthwiseConvolution(dim, kernel_size=7, bias=True, dimension=D)
        self.norm = MinkowskiLayerNorm(dim, 1e-6)
        self.pwconv1 = MinkowskiLinear(dim, 4 * dim)
        self.act = MinkowskiGELU()
        self.pwconv2 = MinkowskiLinear(4 * dim, dim)
        self.grn = MinkowskiGRN(4 * dim)
        self.drop_path = MinkowskiDropPath(drop_path)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x


class BasicLayer(nn.Module):
    """
    Based on Swin Transformer
    https://arxiv.org/abs/2103.14030
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim,
            depth,
            drop_path=None,
            downsample=None,
            is_conv=False,
            D=2
    ):

        super().__init__()
        if drop_path is None:
            drop_path = []
        self.dim = dim
        self.depth = depth

        self.is_conv = is_conv

        self.blocks = nn.ModuleList(
            [nn.Sequential(
                *[Block(dim=dim, drop_path=drop_path[i], D=2) for i in range(depth)]
            )]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, D=D)
        else:
            self.downsample = None

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SparseConvNeXtV2(nn.Module):
    """ Sparse ConvNeXtV2.

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self,
                 in_chans=6,
                 embed_dim=96,
                 depths=[3, 3, 9, 3, ],
                 is_conv_list=[True, True, True, True, ],
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 is_checkpoint=True,
                 D=3, ):
        super().__init__()
        self.depths = depths

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

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
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                downsample=PatchMerging,
                D=D,
            )
            # if is_checkpoint:
                # layer = checkpoint(layer)
                # layer = checkpoint_wrapper(layer)
            self.layers.append(layer)

        self.norm = MinkowskiLayerNorm(embed_dim, eps=1e-6)
        self.conv_after_body = MinkowskiConvolution(embed_dim, embed_dim, kernel_size=3, stride=1,  bias=True,
                                                    dimension=D)
        # self.conv_last = MinkowskiConvolution(embed_dim, 3, kernel_size=3, stride=1,  bias=True, dimension=D)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiDepthwiseConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight, std=.02)
            if isinstance(m, MinkowskiLinear) and m.linear.bias is not None:
                nn.init.constant_(m.linear.bias, 0)

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p). \
            repeat_interleave(scale, axis=1). \
            repeat_interleave(scale, axis=2)

    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        # x = x.permute(0, 3, 1, 2)  # b c h w
        return x

    def forward(self, x, mask):
        mask = mask.unsqueeze(1).type_as(x)
        x *= (1. - mask)

        x = self.patch_embed(x)
        x = to_sparse(x)
        f_x = self.forward_features(x)
        res = self.conv_after_body(f_x) + x
        x = res.dense()[0]
        return x
