# import torch
# import torch.nn as nn
#
# from MinkowskiEngine import (
#     MinkowskiConvolution,
#     MinkowskiDepthwiseConvolution,
#     MinkowskiLinear,
# )
#
# from timm.models.layers import trunc_normal_
#
# from basicsr.archs.my_arch import Block, MYIR
# from basicsr.utils.registry import ARCH_REGISTRY
# # from basicsr.archs.convnextv2_sparse import SparseConvNeXtV2
#
# @ARCH_REGISTRY.register()
# class MYIR2(nn.Module):
#     """ Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone
#     """
#
#     def __init__(
#             self,
#             img_size=128,
#             in_chans=6,
#             depths=[3, 3, 9, 3],
#             embed_dim=48,
#             decoder_depth=1,
#             decoder_embed_dim=48,
#             patch_size=2,
#             mask_ratio=0.0,
#             norm_pix_loss=False,
#     **kwargs):
#         super().__init__()
#
#         # configs
#         self.img_size = img_size
#         self.depths = depths
#         self.patch_size = patch_size
#         self.mask_ratio = mask_ratio
#         self.num_patches = (img_size // patch_size) ** 2
#         self.decoder_embed_dim = decoder_embed_dim
#         self.decoder_depth = decoder_depth
#         self.norm_pix_loss = norm_pix_loss
#
#         # encoder
#         self.encoder = MYIR(
#             in_chans=in_chans, depths=depths, embed_dim=embed_dim,)
#         # decoder
#         self.proj = nn.Conv2d(
#             in_channels=embed_dim,
#             out_channels=decoder_embed_dim,
#             kernel_size=1)
#         # mask tokens
#         self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
#         decoder = [Block(
#             dim=decoder_embed_dim,
#             drop_path=0.) for _ in range(decoder_depth)]
#         self.decoder = nn.Sequential(*decoder)
#         # pred
#         self.pred = nn.Conv2d(
#             in_channels=decoder_embed_dim,
#             out_channels=3,
#             kernel_size=1)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, MinkowskiConvolution):
#             trunc_normal_(m.kernel, std=.02)
#             nn.init.constant_(m.bias, 0)
#         if isinstance(m, MinkowskiDepthwiseConvolution):
#             trunc_normal_(m.kernel)
#             nn.init.constant_(m.bias, 0)
#         if isinstance(m, MinkowskiLinear):
#             trunc_normal_(m.linear.weight)
#             nn.init.constant_(m.linear.bias, 0)
#         if isinstance(m, nn.Conv2d):
#             w = m.weight.data
#             trunc_normal_(w.view([w.shape[0], -1]))
#             nn.init.constant_(m.bias, 0)
#         if isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         if hasattr(self, 'mask_token'):
#             torch.nn.init.normal_(self.mask_token, std=.02)
#
#     def patchify(self, imgs):
#         """
#         imgs: (N, 3, H, W)
#         x: (N, L, patch_size**2 *3)
#         """
#         p = self.patch_size
#         assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
#
#         h = w = imgs.shape[2] // p
#         x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
#         x = torch.einsum('nchpwq->nhwpqc', x)
#         x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
#         return x
#
#     def unpatchify(self, x):
#         """
#         x: (N, L, patch_size**2 *3)
#         imgs: (N, 3, H, W)
#         """
#         p = self.patch_size
#         h = w = int(x.shape[1] ** .5)
#         assert h * w == x.shape[1]
#
#         x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
#         x = torch.einsum('nhwpqc->nchpwq', x)
#         imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
#         return imgs
#
#     def gen_random_mask(self, x, mask_ratio):
#         N = x.shape[0]
#         L = (x.shape[2] // self.patch_size) ** 2
#         len_keep = int(L * (1 - mask_ratio))
#
#         noise = torch.randn(N, L, device=x.device)
#
#         # sort noise for each sample
#         ids_shuffle = torch.argsort(noise, dim=1)
#         ids_restore = torch.argsort(ids_shuffle, dim=1)
#
#         # generate the binary mask: 0 is keep 1 is remove
#         mask = torch.ones([N, L], device=x.device)
#         mask[:, :len_keep] = 0
#         # unshuffle to get the binary mask
#         mask = torch.gather(mask, dim=1, index=ids_restore)
#         mask = self.upsample_mask(mask, 2)
#         return mask
#
#     def upsample_mask(self, mask, scale):
#         assert len(mask.shape) == 2
#         p = int(mask.shape[1] ** .5)
#         return mask.reshape(-1, p, p). \
#             repeat_interleave(scale, axis=1). \
#             repeat_interleave(scale, axis=2)
#
#     def forward_encoder(self, imgs, mask_ratio):
#         # generate random masks
#         mask = self.gen_random_mask(imgs, mask_ratio)
#         # encoding
#         x = self.encoder(imgs, mask)
#         return x, mask
#
#     def forward_decoder(self, x, mask):
#         x = self.proj(x)
#         # append mask token
#         n, c, h, w = x.shape
#         mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
#         mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
#         x = x * (1. - mask) + mask_token * mask
#         # decoding
#         x = self.decoder(x)
#         # pred
#         pred = self.pred(x)
#         return pred
#
#     def forward(self, imgs, mask_ratio=0.3):
#         # single = imgs[:, -3:, :, :]
#
#         x, mask = self.forward_encoder(imgs, mask_ratio)
#         # pred = self.forward_decoder(x, mask) + single
#         return x
#
# if __name__ == '__main__':
#     model = MYIR2()
#     model.cuda()
#     from torchsummary import summary
#
#     summary(model, (6, 128, 128))
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

from basicsr.utils.registry import ARCH_REGISTRY


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
            D=2
    ):

        super().__init__()
        if drop_path is None:
            drop_path = []
        self.dim = dim
        self.depth = depth


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

@ARCH_REGISTRY.register()
class MYIR2(nn.Module):
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
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 is_checkpoint=True,
                 D=2, **kwargs):
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
        self.conv_last = MinkowskiConvolution(embed_dim, 3, kernel_size=3, stride=1,  bias=True, dimension=D)

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

    def forward(self, x):
        single = torch.clone(x[:, -3:, :, :])

        x = self.patch_embed(x)
        x = to_sparse(x)
        f_x = self.forward_features(x)
        res = self.conv_after_body(f_x) + x
        res=self.conv_last(res)
        x = res.dense()[0] + single
        return x

if __name__ == '__main__':

    model = MYIR2(
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 )
    model.cuda()
    from torchsummary import summary
    from thop import profile
    summary(model, (6, 128, 128))
    flops, params = profile(model, inputs=(torch.randn(1,6,128,128).cuda(),))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
