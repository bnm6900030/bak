from fairscale.nn import checkpoint_wrapper
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np
from einops import rearrange
from timm.models.layers import split_attn
from timm.models.layers import mlp
from torch.nn.init import trunc_normal_

from basicsr.utils.registry import ARCH_REGISTRY

from natten.functional import natten2dqkrpb, natten2dav


class HydraNeighborhoodAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 kernel_sizes=[7],  # Array for kernel sizes
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dilations=[1, 2],  # Array of dilations
                 ):
        super().__init__()
        if len(kernel_sizes) == 1 and len(dilations) != 1:
            kernel_sizes = [kernel_sizes[0] for _ in range(len(dilations))]
        elif len(dilations) == 1 and len(kernel_sizes) != 1:
            dilations = [dilations[0] for _ in range(len(kernel_sizes))]
        assert (len(kernel_sizes) == len(
            dilations)), f"Number of kernels ({(kernel_sizes)}) must be the same as number of dilations ({(dilations)})"
        self.num_splits = len(kernel_sizes)
        self.num_heads = num_heads
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations

        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        asserts = []
        for i in range(len(kernel_sizes)):
            asserts.append(kernel_sizes[i] > 1 and kernel_sizes[i] % 2 == 1)
            if asserts[i] == False:
                warnings.warn(f"Kernel_size {kernel_sizes[i]} needs to be >1 and odd")
        assert (all(asserts)), f"Kernel sizes must be >1 AND odd. Got {kernel_sizes}"

        self.window_size = []
        for i in range(len(dilations)):
            self.window_size.append(self.kernel_sizes[i] * self.dilations[i])

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = nn.Sequential(*[nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False),
                                   nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                             bias=False)
                                   ])
        # Needs to be fixed if we want uneven head splits. // is floored
        # division
        if num_heads % len(kernel_sizes) == 0:
            self.rpb = nn.ParameterList(
                [nn.Parameter(torch.zeros(num_heads // self.num_splits, (2 * k - 1), (2 * k - 1))) for k in
                 kernel_sizes])
            self.clean_partition = True
        else:
            warnings.warn(f"Number of partitions ({self.num_splits}) do not " \
                          f"evenly divide the number of heads ({self.num_heads}). " \
                          f"We evenly divide the remainder between the last " \
                          f"heads This may cause unexpected behavior. Your head " \
                          f"partitions look like {self.shapes}")
            raise

        [trunc_normal_(rpb, std=0.02, mean=0.0, a=-2., b=2.) for rpb in self.rpb]
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, Hp, Wp, = x.shape
        H, W = Hp, Wp
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)

        q, k, v = qkv.chunk(3, dim=0)
        q = q.squeeze(0) * self.scale  # b num_head h w head_dim
        k = k.squeeze(0)
        v = v.squeeze(0)

        q = q.chunk(self.num_splits, dim=1)
        k = k.chunk(self.num_splits, dim=1)
        v = v.chunk(self.num_splits, dim=1)

        attention = [natten2dqkrpb(_q, _k, _rpb, _kernel_size, _dilation) \
                     for _q, _k, _rpb, _kernel_size, _dilation in
                     zip(q, k, self.rpb, self.kernel_sizes, self.dilations)]
        attention = [a.softmax(dim=-1) for a in attention]
        attention = [self.attn_drop(a) for a in attention]

        x = [natten2dav(_attn, _v, _k, _d) for _attn, _v, _k, _d in
             zip(attention, v, self.kernel_sizes, self.dilations)]

        x = torch.cat(x, dim=1)
        x = x.permute(0, 1, 4, 2, 3).reshape(B, C, H, W, )
        x = self.proj_drop(self.proj(x))
        return x


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
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
        return out


class MHSA(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 attn_drop=0.,
                 proj_drop=0.,
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        qkv = self.qkv(x).permute(0, 3, 1, 2)

        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.permute(0, 3, 1, 2)


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.attn = Attention(dim, num_heads, bias)
        # self.attn = MHSA(dim, num_heads)
        # self.attn = split_attn.SplitAttn(in_channels=dim)
        self.attn = HydraNeighborhoodAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.ffn = mlp.GatedMlp(dim, hidden_features=dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # x = x + self.ffn(self.norm2(x).permute(0,2,3,1)).permute(0,3,1,2)
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
@ARCH_REGISTRY.register()
class MYIR3(nn.Module):
    def __init__(self,
                 inp_channels=6,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=8,
                 heads=[4, 4, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):
        super().__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed_C = OverlapPatchEmbed(3, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_C = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.down1_2_C = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_C = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.down2_3_C = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level3_C = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.down3_4_C = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

        self.merge_chan = nn.Conv2d(int(dim * 2 ** 4), int(dim * 2 ** 3), kernel_size=1, bias=bias)

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.refinement = checkpoint_wrapper(self.refinement)
        # self.decoder_level1 = checkpoint_wrapper(self.decoder_level1)
        # self.decoder_level2 = checkpoint_wrapper(self.decoder_level2)
        # self.decoder_level3 = checkpoint_wrapper(self.decoder_level3)
        # self.latent = checkpoint_wrapper(self.latent)
        # self.encoder_level3 = checkpoint_wrapper(self.encoder_level3)
        # self.encoder_level2 = checkpoint_wrapper(self.encoder_level2)
        # self.encoder_level1 = checkpoint_wrapper(self.encoder_level1)
        # self.patch_embed=checkpoint_wrapper(self.patch_embed)
        # self.down1_2 = checkpoint_wrapper(self.down1_2)
        # self.down2_3 = checkpoint_wrapper(self.down2_3)
        # self.down3_4 = checkpoint_wrapper(self.down3_4)
        # self.up4_3 = checkpoint_wrapper(self.up4_3)
        # self.up3_2 = checkpoint_wrapper(self.up3_2)
        # self.up2_1 = checkpoint_wrapper(self.up2_1)

    def forward(self, inp_img, C):
        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1_C = self.patch_embed_C(C)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        out_enc_level1_C = self.encoder_level1_C(inp_enc_level1_C)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_enc_level2_C = self.down1_2_C(out_enc_level1_C)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_enc_level2_C = self.encoder_level2_C(inp_enc_level2_C)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3_C = self.down2_3_C(out_enc_level2_C)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_enc_level3_C = self.encoder_level3_C(out_enc_level3_C)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        inp_enc_level4_C = self.down3_4_C(out_enc_level3_C)

        merge_f = torch.cat([inp_enc_level4_C, inp_enc_level4], 1)
        merge_f = self.merge_chan(merge_f)
        latent = self.latent(merge_f)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3_C], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2_C], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1_C], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1_C)
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1, None


class LocalAttention(Attention):
    def __init__(self, dim, num_heads, bias, base_size=None, kernel_size=None, fast_imp=False, train_size=None):
        super().__init__(dim, num_heads, bias)
        self.base_size = base_size
        self.kernel_size = kernel_size
        self.fast_imp = fast_imp
        self.train_size = train_size

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 3, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def _forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        return out

    def _pad(self, x):
        b, c, h, w = x.shape
        k1, k2 = self.kernel_size
        mod_pad_h = (k1 - h % k1) % k1
        mod_pad_w = (k2 - w % k2) % k2
        pad = (mod_pad_w // 2, mod_pad_w - mod_pad_w // 2, mod_pad_h // 2, mod_pad_h - mod_pad_h // 2)
        x = F.pad(x, pad, 'reflect')
        return x, pad

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        if self.fast_imp:
            raise NotImplementedError
            # qkv, pad = self._pad(qkv)
            # b,C,H,W = qkv.shape
            # k1, k2 = self.kernel_size
            # qkv = qkv.reshape(b,C,H//k1, k1, W//k2, k2).permute(0,2,4,1,3,5).reshape(-1,C,k1,k2)
            # out = self._forward(qkv)
            # out = out.reshape(b,H//k1,W//k2,c,k1,k2).permute(0,3,1,4,2,5).reshape(b,c,H,W)
            # out = out[:,:,pad[-2]:pad[-2]+h, pad[0]:pad[0]+w]
        else:
            qkv = self.grids(qkv)  # convert to local windows
            out = self._forward(qkv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2],
                            w=qkv.shape[-1])
            out = self.grids_inverse(out)  # reverse

        out = self.project_out(out)
        return out


from basicsr.archs.local_arch import AvgPool2d


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)

        if isinstance(m, Attention):
            attn = LocalAttention(dim=m.dim, num_heads=m.num_heads, bias=m.bias, base_size=base_size, fast_imp=False,
                                  train_size=train_size)
            setattr(model, n, attn)


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        C = torch.rand((1, 3, 256, 256))
        with torch.no_grad():
            self.forward(imgs, C)


class MYIR3Local(Local_Base, MYIR3):
    def __init__(self, *args, train_size=(1, 6, 256, 256), base_size=None, fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        MYIR3.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        if base_size is None:
            base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    model = MYIR3()
    model.cuda()
    from torchsummary import summary

    print(1)
    summary(model, [(6, 1680, 1120), (3, 1680, 1120)])
