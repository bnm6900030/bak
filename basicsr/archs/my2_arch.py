from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from common.mixed_attn_block_efficient import EfficientMixAttnTransformerBlock, _get_stripe_info
from common.ops import bchw_to_blc, blc_to_bchw, get_relative_coords_table_all, get_relative_position_index_simple, \
    calculate_mask, calculate_mask_all
from common.swin_v1_block import build_last_conv


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, x_size):
        return bchw_to_blc(self.body(blc_to_bchw(x, x_size)))


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, x_size):
        return bchw_to_blc(self.body(blc_to_bchw(x, x_size)))

class ReduceChan(nn.Module):
    def __init__(self, embed_dim):
        super(ReduceChan, self).__init__()
        self.body = nn.Conv2d(int(embed_dim), int(embed_dim/2), kernel_size=1, bias=False)

    def forward(self, x, x_size):
        B, L, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, int(C*2), *x_size)
        x = self.body(x)
        return x.flatten(2).transpose(1, 2)

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

        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = [w // 2 for w in self.window_size]
        self.stripe_size = stripe_size
        self.stripe_groups = stripe_groups
        self.pretrained_window_size = pretrained_window_size
        self.pretrained_stripe_size = pretrained_stripe_size
        self.anchor_window_down_factor = anchor_window_down_factor
        for k, v in self.set_table_index_mask(self.input_resolution).items():
            self.register_buffer(k, v)

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

    def forward(self, x, x_size):
        table_index_mask = self.get_table_index_mask(x.device, x_size)
        res = x
        for blk in self.blocks:
            res = blk(res, x_size, table_index_mask)
        res = bchw_to_blc(self.conv(blc_to_bchw(res, x_size)))

        return res + x

    def flops(self):
        pass

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


@ARCH_REGISTRY.register()
class MYIR2(nn.Module):

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
            # num_heads_window=[3, 3, 3, 3],
            # num_heads_stripe=[3, 3, 3, 3],
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
            local_connection=True,
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
        super(MYIR2, self).__init__()
        # Process the input arguments
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
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

        self.layers = nn.ModuleList()
        self.encoder_level1 = TransformerStage(
            dim=embed_dim,
            input_resolution=self.input_resolution,
            depth=2,
            num_heads_window=3,
            num_heads_stripe=3,
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
                      sum(depths[:0]): sum(depths[: 0 + 1])
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

        self.down1_2 = Downsample(embed_dim)  ## From Level 1 to Level 2
        self.encoder_level2 = TransformerStage(
            dim=embed_dim*2,
            input_resolution=to_2tuple(int(img_size/2)),
            depth=2,
            num_heads_window=3,
            num_heads_stripe=3,
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
                      sum(depths[:1]): sum(depths[: 1 + 1])
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

        self.down2_3 = Downsample(int(embed_dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = TransformerStage(
            dim=embed_dim*4,
            input_resolution=to_2tuple(int(img_size/4)),
            depth=6,
            num_heads_window=3,
            num_heads_stripe=3,
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
                      sum(depths[:2]): sum(depths[: 2 + 1])
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

        self.down3_4 = Downsample(int(embed_dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = TransformerStage(
            dim=embed_dim*8,
            input_resolution=to_2tuple(int(img_size/8)),
            depth=6,
            num_heads_window=3,
            num_heads_stripe=3,
            window_size=to_2tuple(int(window_size/2)),
            stripe_size=to_2tuple(int(window_size/2)),
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
                      sum(depths[:3]): sum(depths[: 3 + 1])
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

        self.up4_3 = Upsample(int(embed_dim * 2 ** 3))  ## From Level 4 to Level 3
        # self.reduce_chan_level3 = nn.Conv2d(int(embed_dim * 2 ** 3), int(embed_dim * 2 ** 2), kernel_size=1, bias=False)
        self.reduce_chan_level3 = ReduceChan(int(embed_dim * 2 ** 3))
        self.decoder_level3 = TransformerStage(
            dim=embed_dim*4,
            input_resolution=to_2tuple(int(img_size/4)),
            depth=6,
            num_heads_window=3,
            num_heads_stripe=3,
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
                      sum(depths[:2]): sum(depths[: 2 + 1])
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

        self.up3_2 = Upsample(int(embed_dim * 2 ** 2))  ## From Level 3 to Level 2
        # self.reduce_chan_level2 = nn.Conv2d(int(embed_dim * 2 ** 2), int(embed_dim * 2 ** 1), kernel_size=1, bias=False)
        self.reduce_chan_level2 = ReduceChan(int(embed_dim * 2 ** 2))
        self.decoder_level2 = TransformerStage(
            dim=embed_dim*2,
            input_resolution=to_2tuple(int(img_size/2)),
            depth=2,
            num_heads_window=3,
            num_heads_stripe=3,
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
                      sum(depths[:1]): sum(depths[: 1 + 1])
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

        self.up2_1 = Upsample(int(embed_dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = ReduceChan(int(embed_dim * 2 ** 1))

        self.decoder_level1 = TransformerStage(
            dim=embed_dim,
            input_resolution=self.input_resolution,
            depth=2,
            num_heads_window=3,
            num_heads_stripe=3,
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
                      sum(depths[:0]): sum(depths[: 0 + 1])
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

        self.refinement = TransformerStage(
            dim=embed_dim,
            input_resolution=self.input_resolution,
            depth=2,
            num_heads_window=3,
            num_heads_stripe=3,
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
                      sum(depths[:0]): sum(depths[: 0 + 1])
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
        x_size = (x.shape[2], x.shape[3])
        x = bchw_to_blc(x)
        x = self.norm_start(x)
        x = self.pos_drop(x)

        # table_index_mask = self.get_table_index_mask(x.device, x_size)
        # for layer in self.layers:
        #     x = layer(x, x_size, table_index_mask)
        out_enc_level1 = self.encoder_level1(x, x_size)

        inp_enc_level2 = self.down1_2(out_enc_level1, x_size )
        out_enc_level2 = self.encoder_level2(inp_enc_level2, (int(x_size[0]/2),int(x_size[1]/2)),)

        inp_enc_level3 = self.down2_3(out_enc_level2, (int(x_size[0]/2),int(x_size[1]/2)))
        out_enc_level3 = self.encoder_level3(inp_enc_level3, (int(x_size[0]/4),int(x_size[1]/4)), )

        inp_enc_level4 = self.down3_4(out_enc_level3, (int(x_size[0]/4),int(x_size[1]/4)))
        latent = self.latent(inp_enc_level4, (int(x_size[0]/8),int(x_size[1]/8)),)

        inp_dec_level3 = self.up4_3(latent, (int(x_size[0]/8),int(x_size[1]/8)))
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3, (int(x_size[0]/4),int(x_size[1]/4)))
        out_dec_level3 = self.decoder_level3(inp_dec_level3, (int(x_size[0]/4),int(x_size[1]/4)),)

        inp_dec_level2 = self.up3_2(out_dec_level3, (int(x_size[0]/4),int(x_size[1]/4)))
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2,  (int(x_size[0]/2),int(x_size[1]/2)))
        out_dec_level2 = self.decoder_level2(inp_dec_level2,  (int(x_size[0]/2),int(x_size[1]/2)),)

        inp_dec_level1 = self.up2_1(out_dec_level2,  (int(x_size[0]/2),int(x_size[1]/2)))
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1,  x_size)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, x_size,)

        out_dec_level1 = self.refinement(out_dec_level1, x_size, )


        x = self.norm_end(out_dec_level1)  # B L C
        x = blc_to_bchw(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        single = x[:, -3:, :, :]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # for image denoising and JPEG compression artifact reduction
        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first
        x = self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, : H * self.upscale, : W * self.upscale] + single

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
