from fairscale.nn import checkpoint_wrapper
import torch.nn as nn

from natten import NeighborhoodAttention2D as NeighborhoodAttention
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import torch

from basicsr.archs.util import DPD
from basicsr.utils.registry import ARCH_REGISTRY


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
    ):

        super().__init__()
        self.dim = dim
        self.depth = depth

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

        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


def resnet_block(in_channels, kernel_size=3, dilation=[1, 1], bias=True, res_num=1):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias, res_num=res_num)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias, res_num):
        super(ResnetBlock, self).__init__()

        self.res_num = res_num
        self.stem = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0],
                          padding=((kernel_size - 1) // 2) * dilation[0], bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1],
                          padding=((kernel_size - 1) // 2) * dilation[1], bias=bias),
            ) for _ in range(res_num)
        ])

    def forward(self, x):

        if self.res_num > 1:
            temp = x

        for i in range(self.res_num):
            xx = self.stem[i](x)
            x = x + xx
            x = F.leaky_relu(x, 0.1, inplace=True)

        if self.res_num > 1:
            x = x + temp

        return x


class GetDME(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.activation = nn.LeakyReLU(0.2, True)
        self.dme = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                                 self.activation,
                                 resnet_block(embed_dim, kernel_size=3, res_num=2),
                                 resnet_block(embed_dim, kernel_size=3, res_num=2),
                                 nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1), )

    def forward(self, x):
        return self.dme(x)


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=320, ):
        super().__init__()
        self.activation = nn.LeakyReLU(0.2, True)

        self.en_layer1_1 = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, padding=1),
            self.activation,
        )
        self.en_layer1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.en_layer1_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.en_layer1_4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))

        self.en_layer2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            self.activation,
        )
        self.en_layer2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.en_layer2_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.en_layer2_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1))

        self.en_layer3_1 = nn.Sequential(
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            self.activation,
        )

    def forward(self, x):
        hx = self.en_layer1_1(x)
        hx = self.activation(self.en_layer1_2(hx) + hx)
        hx = self.activation(self.en_layer1_3(hx) + hx)
        hx = self.activation(self.en_layer1_4(hx) + hx)
        residual_1 = hx
        hx = self.en_layer2_1(hx)
        hx = self.activation(self.en_layer2_2(hx) + hx)
        hx = self.activation(self.en_layer2_3(hx) + hx)
        hx = self.activation(self.en_layer2_4(hx) + hx)
        residual_2 = hx
        hx = self.en_layer3_1(hx)
        return hx, residual_1, residual_2


class Embeddings_output(nn.Module):
    def __init__(self, embed_dim):
        super(Embeddings_output, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.de_layer3_1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 192, kernel_size=4, stride=2, padding=1),
            self.activation,
        )

        self.de_layer2_2 = nn.Sequential(
            nn.Conv2d(192 + 128, 192, kernel_size=1, padding=0),
            self.activation,
        )

        self.de_block = BasicLayer(
            dim=192,
            depth=6,
            num_heads=6,
            kernel_size=7,
            dilations=None,
            mlp_ratio=2,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            downsample=None,
        )

        self.de_layer2_1 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=4, stride=2, padding=1),
            self.activation,
        )

        self.de_layer1_3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.de_layer1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.de_layer1_1 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            self.activation
        )

    def forward(self, x, residual_1, residual_2):
        hx = self.de_layer3_1(x)
        hx = self.de_layer2_2(torch.cat((hx, residual_2), dim=1))
        hx = hx.permute(0, 2, 3, 1)  # b h w c
        hx = self.de_block(hx)
        hx = hx.permute(0, 3, 1, 2)  # b c h w
        hx = self.de_layer2_1(hx)
        hx = self.activation(self.de_layer1_3(torch.cat((hx, residual_1), dim=1)) + hx)
        hx = self.activation(self.de_layer1_2(hx) + hx)
        hx = self.de_layer1_1(hx)

        return hx


@ARCH_REGISTRY.register()
class MYIR3(nn.Module):
    def __init__(
            self,
            in_chans=3,
            embed_dim=320,
            depths=None,
            num_heads=None,
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

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.patch_embed_dme = PatchEmbed(
            in_chans=6,
            embed_dim=embed_dim,
        )

        # disparity map estimator
        self.DME = GetDME(embed_dim)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                # dim=int(embed_dim * 2 ** i_layer),
                dim=embed_dim,
                depth=depths[i_layer],
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
                downsample=None,
            )
            # if is_checkpoint and i_layer %2 ==0:
            layer = checkpoint_wrapper(layer)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.conv_merge = nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1)

        self.decoder = Embeddings_output(embed_dim)
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

    def forward(self, x, single):
        feature_dme, _, _ = self.patch_embed_dme(x)
        feature_dme = self.DME(feature_dme)
        feature_single, residual_1, residual_2 = self.patch_embed(single)
        feature_mix = torch.cat([feature_single, feature_dme], 1)
        feature_mix = self.conv_merge(feature_mix).permute(0, 2, 3, 1)  # b h w c
        feature_single = self.forward_features(feature_mix) + feature_single
        out = self.decoder(feature_single, residual_1, residual_2) + single

        R_wraped_by_dme = DPD(F.interpolate(x[:, :3, :, :], scale_factor=1 / 4, mode='area'), feature_dme,
                              padding_mode='zeros', )

        return out, R_wraped_by_dme


if __name__ == '__main__':
    model = MYIR3(num_heads=[3, 6, 12, 18],
                  embed_dim=252,
                  depths=[2, 2, 6, 2],
                  )
    model.cuda()
    from torchsummary import summary
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # from thop import profile
    summary(model, [[6, 128, 128], [3, 128, 128]])
    # flops, params = profile(model, inputs=(torch.randn(1,6,256,256).cuda(),))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
# Input size (MB): 294912.00
# Forward/backward pass size (MB): 2199023240189.06
# Params size (MB): 65.90
# Estimated Total Size (MB): 2199023535166.96