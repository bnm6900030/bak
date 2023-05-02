from fairscale.nn import checkpoint_wrapper
import torch.nn as nn

from natten import NeighborhoodAttention2D as NeighborhoodAttention
import torch.nn.functional as F
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

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=1e-6)
        self.mlp = FeedForward(dim, 2.66, False)


    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
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
                    NATransformerLayer(
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
                    )
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
class MYIR(nn.Module):
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
            # if is_checkpoint and i_layer %2 ==0:
            layer = checkpoint_wrapper(layer)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, 3, 3, 1, 1, 1)

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

    def forward(self, x):
        single = torch.clone(x[:, -3:, :, :])
        x = self.patch_embed(x)
        f_x = self.forward_features(x)
        res = self.conv_after_body(f_x) + x.permute(0, 3, 1, 2)
        x = self.conv_last(res) + single
        return x



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
    model = MYIR(num_heads=[3, 6, 12, 24],
                 embed_dim=48,
                 depths=[2, 2, 6, 2 ],
                 is_conv_list=[False, False,False,False, ],
                 )
    model.cuda()
    from torchsummary import summary
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # from thop import profile
    summary(model, (6, 1120,880))
    # flops, params = profile(model, inputs=(torch.randn(1,6,256,256).cuda(),))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')

