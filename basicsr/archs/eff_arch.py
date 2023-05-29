import itertools

from einops import rearrange
from fairscale.nn import checkpoint_wrapper
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
import torch

from basicsr.archs.utils import LayerNorm
from basicsr.utils.registry import ARCH_REGISTRY


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio, act_layer=nn.GELU, drop=0., mid_conv=False):
        super().__init__()
        self.mid_conv = mid_conv
        hidden_features = dim * mlp_ratio
        self.act = act_layer()
        self.fc1 = nn.Conv2d(in_channels=dim, out_channels=hidden_features, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels=hidden_features // 2, out_channels=dim, kernel_size=1)
        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features // 2, hidden_features // 2, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features // 2)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # bhwc 2 bchw

        x = self.fc1(x)
        x = self.act(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        if self.mid_conv:
            x_mid = self.mid(x)
            x = self.act(x_mid)
        x = self.fc2(x) * self.beta
        x = self.drop(x)
        x = x.permute(0, 2, 3, 1)  # bchw 2 bhwc
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Attention(nn.Module):
    def __init__(self, dim=384, num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 act_layer=nn.ReLU):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        # self.attn_ratio = attn_ratio
        self.scale = head_dim ** -0.5
        # self.dh = attn_ratio * dim

        self.resolution = resolution
        self.N = self.resolution ** 2
        self.q = nn.Sequential(nn.Conv2d(dim, dim, 1), )
        self.k = nn.Sequential(nn.Conv2d(dim, dim, 1), )
        self.v = nn.Sequential(nn.Conv2d(dim, dim, 1), )
        self.v_local = nn.Sequential(nn.Conv2d(dim, dim,
                                               kernel_size=3, stride=1, padding=1, groups=dim), )

        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)

        self.proj = nn.Sequential(act_layer(),
                                  nn.Conv2d(dim, dim, 1), )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # bhwc 2 bchw

        B, C, H, W = x.shape
        q = rearrange(self.q(x), 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(memory_format=torch.contiguous_format)
        k = rearrange(self.k(x), 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(memory_format=torch.contiguous_format)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        v = self.v(x)
        v_local = self.v_local(v)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(memory_format=torch.contiguous_format)


        attn = ((q @ k.transpose(-2, -1)) * self.scale)
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v)

        out = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W) + v_local

        out = self.proj(out)
        out = out.permute(0, 2, 3, 1)  # bchw 2 bhwc
        return out


class AttnFFN(nn.Module):
    def __init__(
            self,
            dim, mlp_ratio=2,
            attn_ratio=4,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            drop=0.,
            drop_path=0.,
            resolution=7,
            num_heads=8
    ):
        super().__init__()
        self.token_mixer = Attention(dim, attn_ratio=attn_ratio, resolution=resolution, num_heads=num_heads)
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.norm2 = norm_layer(dim, eps=1e-6)
        self.mlp = Mlp(dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, drop=drop, mid_conv=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp((self.norm2(x))))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
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
            mlp_ratio=2,
            attn_ratio=4,
            drop=0.0,
            drop_path=None,
            norm_layer=nn.LayerNorm,
            downsample=None,
            resolution=7,
    ):

        super().__init__()
        if drop_path is None:
            drop_path = []
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                AttnFFN(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    resolution=resolution,
                    drop=drop,
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


@ARCH_REGISTRY.register()
class Eff(nn.Module):
    def __init__(
            self,
            in_chans=6,
            img_size=256,
            embed_dim=48,
            depths=None,
            num_heads=None,
            mlp_ratio=2,
            attn_ratio=4,
            drop_rate=0.0,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            is_checkpoint=True,
            **kwargs
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.attn_ratio = attn_ratio

        # split image into non-overlapping patches
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        # self.patch_embed = checkpoint_wrapper(self.patch_embed)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=self.mlp_ratio,
                attn_ratio=self.attn_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging,
                resolution=img_size,
            )
            # if i_layer % 2 == 0:
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

    def forward_features(self, x):
        x = x.permute(0, 2, 3, 1)  # b h w c
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # b c h w
        return x

    def forward(self, x, C):
        x = self.patch_embed(x)
        f_x = self.forward_features(x)
        res = self.conv_after_body(f_x) + x
        x = self.conv_last(res) + C
        return x, None


if __name__ == '__main__':
    model = Eff(num_heads=[2, 6, 8, 12,8,8],
                embed_dim=48,
                depths=[2, 2, 2, 2,2,2],
                )
    model.cuda()
    from torchsummary import summary
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    summary(model, [(6, 256, 256), (3, 256, 256)])
