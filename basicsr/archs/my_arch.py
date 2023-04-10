from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from common.mixed_attn_block_efficient import EfficientMixAttnTransformerBlock
from common.ops import bchw_to_blc, blc_to_bchw, get_relative_coords_table_all, get_relative_position_index_simple, \
    calculate_mask, calculate_mask_all
from common.swin_v1_block import build_last_conv


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


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


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).cuda())).exp()
        # logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.ffn = FeedForward(dim, 2.66, False)

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        # x = x + self.drop_path(self.norm2(self.mlp(x)))
        x = x + self.ffn(self.norm2(x))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class TransformerStage(nn.Module):
    """Transformer stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads_window (list[int]): Number of window attention heads in different layers.
        num_heads_stripe (list[int]): Number of stripe attention heads in different layers.
        stripe_size (list[int]): Stripe size. Default: [8, 8]
        stripe_groups (list[int]): Number of stripe groups. Default: [None, None].
        stripe_shift (bool): whether to shift the stripes. This is used as an ablation study.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qkv_proj_type (str): QKV projection type. Default: linear. Choices: linear, separable_conv.
        anchor_proj_type (str): Anchor projection type. Default: avgpool. Choices: avgpool, maxpool, conv2d, separable_conv, patchmerging.
        anchor_one_stage (bool): Whether to use one operator or multiple progressive operators to reduce feature map resolution. Default: True.
        anchor_window_down_factor (int): The downscale factor used to get the anchors.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        pretrained_window_size (list[int]): pretrained window size. This is actually not used. Default: [0, 0].
        pretrained_stripe_size (list[int]): pretrained stripe size. This is actually not used. Default: [0, 0].
        conv_type: The convolutional block before residual connection.
        init_method: initialization method of the weight parameters used to train large scale models.
            Choices: n, normal -- Swin V1 init method.
                    l, layernorm -- Swin V2 init method. Zero the weight and bias in the post layer normalization layer.
                    r, res_rescale -- EDSR rescale method. Rescale the residual blocks with a scaling factor 0.1
                    w, weight_rescale -- MSRResNet rescale method. Rescale the weight parameter in residual blocks with a scaling factor 0.1
                    t, trunc_normal_ -- nn.Linear, trunc_normal; nn.Conv2d, weight_rescale
        fairscale_checkpoint (bool): Whether to use fairscale checkpoint.
        offload_to_cpu (bool): used by fairscale_checkpoint
        args:
            out_proj_type (str): Type of the output projection in the self-attention modules. Default: linear. Choices: linear, conv2d.
            local_connection (bool): Whether to enable the local modelling module (two convs followed by Channel attention). For GRL base model, this is used.                "local_connection": local_connection,
            euclidean_dist (bool): use Euclidean distance or inner product as the similarity metric. An ablation study.
    """

    def __init__(
            self,
            dim,
            input_resolution,
            depth,
            num_heads_window,
            num_heads_stripe,
            window_size,
            stripe_size,
            stripe_groups,
            stripe_shift,
            mlp_ratio=4.0,
            qkv_bias=True,
            qkv_proj_type="linear",
            anchor_proj_type="avgpool",
            anchor_one_stage=True,
            anchor_window_down_factor=1,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            pretrained_window_size=[0, 0],
            pretrained_stripe_size=[0, 0],
            conv_type="1conv",
            init_method="",
            fairscale_checkpoint=False,
            offload_to_cpu=False,
            args=None,
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.init_method = init_method

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = EfficientMixAttnTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads_w=num_heads_window,
                num_heads_s=num_heads_stripe,
                window_size=window_size,
                window_shift=i % 2 == 0,
                stripe_size=stripe_size,
                stripe_groups=stripe_groups,
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift=i % 4 in [2, 3] if stripe_shift else False,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type=qkv_proj_type,
                anchor_proj_type=anchor_proj_type,
                anchor_one_stage=anchor_one_stage,
                anchor_window_down_factor=anchor_window_down_factor,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                pretrained_stripe_size=pretrained_stripe_size,
                res_scale=0.1 if init_method == "r" else 1.0,
                args=args,
            )
            # print(fairscale_checkpoint, offload_to_cpu)
            if fairscale_checkpoint:
                block = checkpoint_wrapper(block, offload_to_cpu=offload_to_cpu)
            self.blocks.append(block)

        self.conv = build_last_conv(conv_type, dim)

    def _init_weights(self):
        for n, m in self.named_modules():
            if self.init_method == "w":
                if isinstance(m, (nn.Linear, nn.Conv2d)) and n.find("cpb_mlp") < 0:
                    print("nn.Linear and nn.Conv2d weight initilization")
                    m.weight.data *= 0.1
            elif self.init_method == "l":
                if isinstance(m, nn.LayerNorm):
                    print("nn.LayerNorm initialization")
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 0)
            elif self.init_method.find("t") >= 0:
                scale = 0.1 ** (len(self.init_method) - 1) * int(self.init_method[-1])
                if isinstance(m, nn.Linear) and n.find("cpb_mlp") < 0:
                    trunc_normal_(m.weight, std=scale)
                elif isinstance(m, nn.Conv2d):
                    m.weight.data *= 0.1
                print(
                    "Initialization nn.Linear - trunc_normal; nn.Conv2d - weight rescale."
                )
            else:
                raise NotImplementedError(
                    f"Parameter initialization method {self.init_method} not implemented in TransformerStage."
                )

    def forward(self, x, x_size, table_index_mask):
        res = x
        for blk in self.blocks:
            res = blk(res, x_size, table_index_mask)
        res = bchw_to_blc(self.conv(blc_to_bchw(res, x_size)))

        return res + x

    def flops(self):
        pass


@ARCH_REGISTRY.register()
class MYIR(nn.Module):

    def __init__(
        self,
        img_size=64,
        in_channels=6,
        out_channels=None,
        embed_dim=64,
        upscale=2,
        img_range=1.0,
        upsampler="",
        depths=[6, 6, 6, 6],
        num_heads_window=[3, 3, 3, 3],
        num_heads_stripe=[3, 3, 3, 3],
        window_size=8,
        stripe_size=[8, 8],  # used for stripe window attention
        stripe_groups=[None, None],
        stripe_shift=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        out_proj_type="linear",
        local_connection=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        conv_type="1conv",
        init_method="n",  # initialization method of the weight parameters used to train large scale models.
        fairscale_checkpoint=False,  # fairscale activation checkpointing
        offload_to_cpu=False,
        euclidean_dist=False,
        **kwargs,
    ):
        super(MYIR, self).__init__()
        # Process the input arguments
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_out_feats = 64
        self.embed_dim = embed_dim
        self.upscale = upscale
        self.upsampler = upsampler
        self.img_range = img_range
        if in_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        max_stripe_size = max([0 if s is None else s for s in stripe_size])
        max_stripe_groups = max([0 if s is None else s for s in stripe_groups])
        max_stripe_groups *= anchor_window_down_factor
        self.pad_size = max(window_size, max_stripe_size, max_stripe_groups)
        # if max_stripe_size >= window_size:
        #     self.pad_size *= anchor_window_down_factor
        # if stripe_groups[0] is None and stripe_groups[1] is None:
        #     self.pad_size = max(stripe_size)
        # else:
        #     self.pad_size = window_size
        self.input_resolution = to_2tuple(img_size)
        self.window_size = to_2tuple(window_size)
        self.shift_size = [w // 2 for w in self.window_size]
        self.stripe_size = stripe_size
        self.stripe_groups = stripe_groups
        self.pretrained_window_size = pretrained_window_size
        self.pretrained_stripe_size = pretrained_stripe_size
        self.anchor_window_down_factor = anchor_window_down_factor

        # Head of the network. First convolution.
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # Body of the network
        self.norm_start = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # stochastic depth decay rule
        args = OmegaConf.create(
            {
                "out_proj_type": out_proj_type,
                "local_connection": local_connection,
                "euclidean_dist": euclidean_dist,
            }
        )
        for k, v in self.set_table_index_mask(self.input_resolution).items():
            self.register_buffer(k, v)

        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = TransformerStage(
                dim=embed_dim,
                input_resolution=self.input_resolution,
                depth=depths[i],
                num_heads_window=num_heads_window[i],
                num_heads_stripe=num_heads_stripe[i],
                window_size=self.window_size,
                stripe_size=stripe_size,
                stripe_groups=stripe_groups,
                stripe_shift=stripe_shift,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type=qkv_proj_type,
                anchor_proj_type=anchor_proj_type,
                anchor_one_stage=anchor_one_stage,
                anchor_window_down_factor=anchor_window_down_factor,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i]) : sum(depths[: i + 1])
                ],  # no impact on SR results
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                pretrained_stripe_size=pretrained_stripe_size,
                conv_type=conv_type,
                init_method=init_method,
                fairscale_checkpoint=fairscale_checkpoint,
                offload_to_cpu=offload_to_cpu,
                args=args,
            )
            self.layers.append(layer)
        self.norm_end = norm_layer(embed_dim)

        # Tail of the network
        self.conv_after_body = build_last_conv(conv_type, embed_dim)

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        self.conv_last = nn.Conv2d(embed_dim, 3, 3, 1, 1)

        self.apply(self._init_weights)
        if init_method in ["l", "w"] or init_method.find("t") >= 0:
            for layer in self.layers:
                layer._init_weights()

    def set_table_index_mask(self, x_size):
        """
        Two used cases:
        1) At initialization: set the shared buffers.
        2) During forward pass: get the new buffers if the resolution of the input changes
        """
        # ss - stripe_size, sss - stripe_shift_size
        ss, sss = _get_stripe_info(self.stripe_size, self.stripe_groups, True, x_size)
        df = self.anchor_window_down_factor

        table_w = get_relative_coords_table_all(
            self.window_size, self.pretrained_window_size
        )
        table_sh = get_relative_coords_table_all(ss, self.pretrained_stripe_size, df)
        table_sv = get_relative_coords_table_all(
            ss[::-1], self.pretrained_stripe_size, df
        )

        index_w = get_relative_position_index_simple(self.window_size)
        index_sh_a2w = get_relative_position_index_simple(ss, df, False)
        index_sh_w2a = get_relative_position_index_simple(ss, df, True)
        index_sv_a2w = get_relative_position_index_simple(ss[::-1], df, False)
        index_sv_w2a = get_relative_position_index_simple(ss[::-1], df, True)

        mask_w = calculate_mask(x_size, self.window_size, self.shift_size)
        mask_sh_a2w = calculate_mask_all(x_size, ss, sss, df, False)
        mask_sh_w2a = calculate_mask_all(x_size, ss, sss, df, True)
        mask_sv_a2w = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, False)
        mask_sv_w2a = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, True)
        return {
            "table_w": table_w,
            "table_sh": table_sh,
            "table_sv": table_sv,
            "index_w": index_w,
            "index_sh_a2w": index_sh_a2w,
            "index_sh_w2a": index_sh_w2a,
            "index_sv_a2w": index_sv_a2w,
            "index_sv_w2a": index_sv_w2a,
            "mask_w": mask_w,
            "mask_sh_a2w": mask_sh_a2w,
            "mask_sh_w2a": mask_sh_w2a,
            "mask_sv_a2w": mask_sv_a2w,
            "mask_sv_w2a": mask_sv_w2a,
        }

    def get_table_index_mask(self, device=None, input_resolution=None):
        # Used during forward pass
        if input_resolution == self.input_resolution:
            return {
                "table_w": self.table_w,
                "table_sh": self.table_sh,
                "table_sv": self.table_sv,
                "index_w": self.index_w,
                "index_sh_a2w": self.index_sh_a2w,
                "index_sh_w2a": self.index_sh_w2a,
                "index_sv_a2w": self.index_sv_a2w,
                "index_sv_w2a": self.index_sv_w2a,
                "mask_w": self.mask_w,
                "mask_sh_a2w": self.mask_sh_a2w,
                "mask_sh_w2a": self.mask_sh_w2a,
                "mask_sv_a2w": self.mask_sv_a2w,
                "mask_sv_w2a": self.mask_sv_w2a,
            }
        else:
            table_index_mask = self.set_table_index_mask(input_resolution)
            for k, v in table_index_mask.items():
                table_index_mask[k] = v.to(device)
            return table_index_mask

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Only used to initialize linear layers
            # weight_shape = m.weight.shape
            # if weight_shape[0] > 256 and weight_shape[1] > 256:
            #     std = 0.004
            # else:
            #     std = 0.02
            # print(f"Standard deviation during initialization {std}.")
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.pad_size - h % self.pad_size) % self.pad_size
        mod_pad_w = (self.pad_size - w % self.pad_size) % self.pad_size
        # print("padding size", h, w, self.pad_size, mod_pad_h, mod_pad_w)

        try:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        except BaseException:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant")
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = bchw_to_blc(x)
        x = self.norm_start(x)
        x = self.pos_drop(x)

        table_index_mask = self.get_table_index_mask(x.device, x_size)
        for layer in self.layers:
            x = layer(x, x_size, table_index_mask)

        x = self.norm_end(x)  # B L C
        x = blc_to_bchw(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        # x = self.check_image_size(x)
        single = x[:, -3:, :, :]
        x = x[:, :6, :, :]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range


        # for image denoising and JPEG compression artifact reduction
        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first
        # if self.in_channels == self.out_channels:
        #     x = x + self.conv_last(res)
        # else:
        x = self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, : H * self.upscale, : W * self.upscale]+single

    def flops(self):
        pass

    def convert_checkpoint(self, state_dict):
        for k in list(state_dict.keys()):
            if (
                k.find("relative_coords_table") >= 0
                or k.find("relative_position_index") >= 0
                or k.find("attn_mask") >= 0
                or k.find("model.table_") >= 0
                or k.find("model.index_") >= 0
                or k.find("model.mask_") >= 0
                # or k.find(".upsample.") >= 0
            ):
                state_dict.pop(k)
                print(k)
        return state_dict
