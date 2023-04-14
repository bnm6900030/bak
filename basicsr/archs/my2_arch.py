import numbers

import torch
import torch.nn as nn
from einops import rearrange

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from basicsr.utils.registry import ARCH_REGISTRY

from timm.models.layers import to_2tuple, trunc_normal_

from common.mixed_attn_block_efficient import EfficientMixAttnTransformerBlock
from common.swin_v1_block import build_last_conv
from fairscale.nn import checkpoint_wrapper


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


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


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(

            # nn.Conv2d(n_feat, n_feat // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
            # nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False,),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            # nn.Conv2d(n_feat, n_feat * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
            # nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False,),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class ReduceChan(nn.Module):
    def __init__(self, embed_dim):
        super(ReduceChan, self).__init__()
        self.body = nn.Conv2d(int(embed_dim), int(embed_dim / 2), kernel_size=1, bias=False)

    def forward(self, x):
        return self.body(x)


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
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            conv_type="1conv",
            init_method="",

    ):
        super().__init__()

        self.dim = dim
        self.init_method = init_method

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = EfficientMixAttnTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,

                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                res_scale=0.1 if init_method == "r" else 1.0,
            )

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

    def forward(self, x):
        res = x
        for blk in self.blocks:
            res = blk(res)
        res = self.conv(res)

        return res + x

    def flops(self):
        pass


@ARCH_REGISTRY.register()
class MYIR2(nn.Module):

    def __init__(
            self,
            img_size=128,
            in_chans=6,
            embed_dim=48,
            upscale=1,
            img_range=1.0,
            upsampler="",
            depths=[2, 2, 6, 2, 6, 2, 2, 2],
            mlp_ratio=2.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=LayerNorm,
            conv_type="1conv",
            init_method="n",  # initialization method of the weight parameters used to train large scale models.
            fairscale_checkpoint=False,
            **kwargs,
    ):
        super(MYIR2, self).__init__()
        # Process the input arguments
        self.in_channels = in_chans
        self.embed_dim = embed_dim
        self.upscale = upscale
        self.upsampler = upsampler
        self.img_range = img_range

        self.input_resolution = to_2tuple(img_size)
        # Head of the network. First convolution.
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Body of the network
        self.norm_start = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        self.encoder_level1 = TransformerStage(
            dim=embed_dim,
            input_resolution=self.input_resolution,
            depth=depths[0],
            mlp_ratio=mlp_ratio,

            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:0]): sum(depths[: 0 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            conv_type=conv_type,
            init_method=init_method,
        )

        self.down1_2 = Downsample(embed_dim)  ## From Level 1 to Level 2

        self.encoder_level2 = TransformerStage(
            dim=embed_dim * 2,
            input_resolution=to_2tuple(int(img_size / 2)),
            depth=depths[1],
            mlp_ratio=mlp_ratio,

            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:1]): sum(depths[: 1 + 1])],
            norm_layer=norm_layer,
            conv_type=conv_type,
            init_method=init_method,
        )

        self.down2_3 = Downsample(int(embed_dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = TransformerStage(
            dim=embed_dim * 4,
            input_resolution=to_2tuple(int(img_size / 4)),
            depth=depths[2],

            mlp_ratio=mlp_ratio,

            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:2]): sum(depths[: 2 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            conv_type=conv_type,
            init_method=init_method,
        )

        self.down3_4 = Downsample(int(embed_dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = TransformerStage(
            dim=embed_dim * 8,
            input_resolution=to_2tuple(int(img_size / 8)),
            depth=depths[3],
            mlp_ratio=mlp_ratio,

            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:3]): sum(depths[: 3 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            conv_type=conv_type,
            init_method=init_method,
        )

        self.up4_3 = Upsample(int(embed_dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = ReduceChan(int(embed_dim * 2 ** 3))

        self.decoder_level3 = TransformerStage(
            dim=embed_dim * 4,
            input_resolution=to_2tuple(int(img_size / 4)),
            depth=depths[4],
            mlp_ratio=mlp_ratio,

            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:4]): sum(depths[: 4 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            conv_type=conv_type,
            init_method=init_method,
        )

        self.up3_2 = Upsample(int(embed_dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = ReduceChan(int(embed_dim * 2 ** 2))

        self.decoder_level2 = TransformerStage(
            dim=embed_dim * 2,
            input_resolution=to_2tuple(int(img_size / 2)),
            depth=depths[5],
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[
                      sum(depths[:5]): sum(depths[: 5 + 1])
                      ],  # no impact on SR results
            norm_layer=norm_layer,
            conv_type=conv_type,
            init_method=init_method,

        )

        self.up2_1 = Upsample(int(embed_dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = ReduceChan(int(embed_dim * 2 ** 1))

        self.decoder_level1 = TransformerStage(
            dim=embed_dim,
            input_resolution=self.input_resolution,
            depth=depths[6],
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:6]): sum(depths[: 6 + 1])],
            norm_layer=norm_layer,
            conv_type=conv_type,
            init_method=init_method,
        )

        self.refinement = TransformerStage(
            dim=embed_dim,
            input_resolution=self.input_resolution,
            depth=depths[7],
            mlp_ratio=mlp_ratio,

            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:7]): sum(depths[:7 + 1])],
            norm_layer=norm_layer,
            conv_type=conv_type,
            init_method=init_method,
        )

        if fairscale_checkpoint:
            self.encoder_level1 = checkpoint_wrapper(self.encoder_level1)
            self.encoder_level2 = checkpoint_wrapper(self.encoder_level2)
            self.encoder_level3 = checkpoint_wrapper(self.encoder_level3)
            self.down1_2 = checkpoint_wrapper(self.down1_2)
            self.down2_3 = checkpoint_wrapper(self.down2_3)
            self.down3_4 = checkpoint_wrapper(self.down3_4)
            self.latent = checkpoint_wrapper(self.latent)
            self.up4_3 = checkpoint_wrapper(self.up4_3)
            self.up3_2 = checkpoint_wrapper(self.up3_2)
            self.up2_1 = checkpoint_wrapper(self.up2_1)
            self.reduce_chan_level1 = checkpoint_wrapper(self.reduce_chan_level1)
            self.reduce_chan_level2 = checkpoint_wrapper(self.reduce_chan_level2)
            self.decoder_level3 = checkpoint_wrapper(self.decoder_level3)
            self.decoder_level2 = checkpoint_wrapper(self.decoder_level2)
            self.decoder_level1 = checkpoint_wrapper(self.decoder_level1)
            self.refinement = checkpoint_wrapper(self.refinement)
        self.norm_end = norm_layer(embed_dim)

        # Tail of the network
        self.conv_after_body = build_last_conv(conv_type, embed_dim)

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        self.conv_last = nn.Conv2d(embed_dim, 3, 3, 1, 1, 1)

        self.apply(self._init_weights)
        if init_method in ["l", "w"] or init_method.find("t") >= 0:
            for layer in self.layers:
                layer._init_weights()

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

    def forward_features(self, x):
        x = self.norm_start(x)
        x = self.pos_drop(x)

        out_enc_level1 = self.encoder_level1(x)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2, )
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3, )
        latent = self.latent(inp_enc_level4, )

        inp_dec_level3 = self.up4_3(latent, )
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3, )
        out_dec_level3 = self.decoder_level3(inp_dec_level3, )

        inp_dec_level2 = self.up3_2(out_dec_level3, )
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2, )
        out_dec_level2 = self.decoder_level2(inp_dec_level2, )

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, )

        out_dec_level1 = self.refinement(out_dec_level1, )

        x = self.norm_end(out_dec_level1)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        single = x[:, -3:, :, :]
        # coc = single - x[:, :3, :, :]
        # coc = self.project_in(coc)
        # coc1, coc2 = self.dwconv(coc).chunk(2, dim=1)
        # coc = F.gelu(coc1) * coc2
        # coc = self.project_out(coc)
        # x = single
        # self.mean = self.mean.type_as(x)
        # x = (x - self.mean) * self.img_range

        # for image denoising and JPEG compression artifact reduction
        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first
        x = self.conv_last(res) + single

        # x = x / self.img_range + self.mean
        # x = x*0.001 + single + coc* self.temperature*0.001
        # x = coc* self.temperature + single
        return x[:, :, : H * self.upscale, : W * self.upscale]

    def flops(self):
        pass


if __name__ == '__main__':
    model = MYIR2()
    model.cuda()
    from torchsummary import summary

    summary(model, (6, 128, 128))
