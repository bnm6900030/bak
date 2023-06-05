from einops import rearrange
from fairscale.nn import checkpoint_wrapper
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
import torch
from natten import NeighborhoodAttention2D as NeighborhoodAttention

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
    def __init__(self, dim, num_heads, bias, kernel_size=7):
        super(Attention, self).__init__()
        self.dim = dim
        self.bias = bias

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x=x.permute(0, 3, 1, 2)  # bhwc 2 bchw
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = out.permute(0, 2, 3, 1)  # bchw 2 bhwc

        return out

class AttnFFN(nn.Module):
    def __init__(
            self,
            dim, mlp_ratio=2,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            drop=0.,
            drop_path=0.,
            num_heads=4,
            kernel_size=7,
            dilation=1,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
    ):
        super().__init__()
        self.token_mixer = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.norm2 = norm_layer(dim, eps=1e-6)
        self.mlp = Mlp(dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, drop=drop, mid_conv=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp((self.norm2(x))))
        return x



@ARCH_REGISTRY.register()
class Eff(nn.Module):
    def __init__(
            self,
            in_chans=6,
            embed_dim=48,
            mlp_ratio=2,
            middle_blk_num=1, enc_blk_nums=[1, 1, 28], dec_blk_nums=[ 1, 1, 1],
            is_checkpoint=True,
            **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        # self.patch_embed = checkpoint_wrapper(self.patch_embed)
        # build layers
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[AttnFFN(embed_dim) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(embed_dim, 2*embed_dim, 2, 2)
            )
            embed_dim = embed_dim * 2

        self.middle_blks = nn.Sequential(
                *[AttnFFN(embed_dim) for _ in range(middle_blk_num)]
            )
        # self.middle_blks = checkpoint_wrapper(self.middle_blks)
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            embed_dim = embed_dim // 2
            self.decoders.append(
                nn.Sequential(
                    *[AttnFFN(embed_dim) for _ in range(num)]
                )
            )

        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.conv_after_body = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(self.embed_dim, 3, 3, 1, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = x.permute(0, 2, 3, 1)  # b h w c

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + enc_skip
            x = decoder(x)
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
    model = Eff( embed_dim=48,
                )
    model.cuda()
    from torchsummary import summary
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    summary(model, [(6, 256, 256), (3, 256, 256)])
