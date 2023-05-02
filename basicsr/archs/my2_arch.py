import torch
import torch.nn as nn
import torch.nn.functional as Func
import collections
from basicsr.utils.registry import ARCH_REGISTRY
from fairscale.nn import checkpoint_wrapper

# from basicsr.archs.util import DPD


def conv(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True, act='LeakyReLU'):
    if act is not None:
        if act == 'LeakyReLU':
            act_ = nn.LeakyReLU(0.1, inplace=True)
        elif act == 'Sigmoid':
            act_ = nn.Sigmoid()
        elif act == 'Tanh':
            act_ = nn.Tanh()

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) // 2) * dilation, bias=bias),
            act_
        )
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                         padding=((kernel_size - 1) // 2) * dilation, bias=bias)


def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


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
            x = Func.leaky_relu(x, 0.1, inplace=True)

        if self.res_num > 1:
            x = x + temp

        return x


def IAC(feat_in, F, N, c, k, is_act_last=True):
    Fs = torch.split(F[:, :N * (c * k * 2), :, :], c * k * 2, dim=1)
    F_bs = torch.split(F[:, N * (c * k * 2):, :, :], c, dim=1)
    for i in range(N):
        F1, _ = torch.split(Fs[i], c * k, dim=1)
        f = SAC(feat_in=feat_in if i == 0 else f, kernel1=F1, ksize=k)  ## image
        f = f + F_bs[i]

        if i < (N - 1):
            f = Func.leaky_relu(f, 0.1, inplace=True)
        elif is_act_last:
            f = Func.leaky_relu(f, 0.1, inplace=True)

    return f


def SAC(feat_in, kernel1, ksize):
    channels = feat_in.size(1)
    N, kernels, H, W = kernel1.size()
    pad = (ksize - 1) // 2

    feat_in = Func.pad(feat_in, (0, 0, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 4).view(N, H, W, channels, -1)

    kernel1 = kernel1.permute(0, 2, 3, 1).view(N, H, W, channels, ksize)
    feat_in = torch.sum(torch.mul(feat_in, kernel1), -1)
    feat_in = feat_in.permute(0, 3, 1, 2)

    feat_in = Func.pad(feat_in, (pad, pad, 0, 0), mode="replicate")
    feat_in = feat_in.unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 4).view(N, H, W, channels, -1)
    # kernel2 = kernel2.permute(0, 2, 3, 1).view(N, H, W, channels, ksize)
    feat_in = torch.sum(torch.mul(feat_in, kernel1), -1)
    feat_out = feat_in.permute(0, 3, 1, 2)

    return feat_out


@ARCH_REGISTRY.register()
class MYIR2(nn.Module):
    def __init__(self, ch=32, ks=3, Fs=3, res_num=2, N=17, wiF=1.5, ):
        super(MYIR2, self).__init__()
        self.device = 'cuda'
        ks = ks
        self.Fs = Fs
        res_num = res_num

        ch1 = ch
        ch2 = ch1 * 2
        ch3 = ch1 * 4
        ch4 = ch1 * 4
        self.ch4 = ch4

        # weight init for filter predictor
        self.wiF = wiF

        ###################################
        # Feature Extractor - Reconstructor
        ###################################
        # feature extractor
        self.conv1_1 = conv(3, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.conv1_3 = conv(ch1, ch1, kernel_size=ks, stride=1)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.conv2_3 = conv(ch2, ch2, kernel_size=ks, stride=1)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.conv3_3 = conv(ch3, ch3, kernel_size=ks, stride=1)

        self.conv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.conv4_2 = conv(ch4, ch4, kernel_size=ks, stride=1)
        self.conv4_3 = conv(ch4, ch4, kernel_size=ks, stride=1)

        self.conv4_4 = nn.Sequential(
            conv(2 * ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            conv(ch4, ch4, kernel_size=ks))
        # reconstructor
        self.conv_res = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=3),
            conv(ch4, ch4, kernel_size=ks))

        self.upconv3_u = upconv(ch4, ch3)
        self.upconv3_1 = resnet_block(ch3, kernel_size=ks, res_num=1)
        self.upconv3_2 = resnet_block(ch3, kernel_size=ks, res_num=1)

        self.upconv2_u = upconv(ch3, ch2)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks, res_num=1)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks, res_num=1)

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_1 = resnet_block(ch1, kernel_size=ks, res_num=1)
        self.upconv1_2 = resnet_block(ch1, kernel_size=ks, res_num=1)

        self.out_res = conv(ch1, 3, kernel_size=ks)
        ###################################

        ###################################
        # IFAN
        ###################################
        # filter encoder
        self.kconv1_1 = conv(6, ch1, kernel_size=ks, stride=1)
        self.kconv1_2 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.kconv1_3 = conv(ch1, ch1, kernel_size=ks, stride=1)

        self.kconv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.kconv2_2 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.kconv2_3 = conv(ch2, ch2, kernel_size=ks, stride=1)

        self.kconv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.kconv3_2 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.kconv3_3 = conv(ch3, ch3, kernel_size=ks, stride=1)

        self.kconv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.kconv4_2 = conv(ch4, ch4, kernel_size=ks, stride=1)
        self.kconv4_3 = conv(ch4, ch4, kernel_size=ks, stride=1)

        # disparity map estimator
        self.DME = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            conv(ch4, 1, kernel_size=3, act=None))

        # filter predictor
        self.conv_DME = conv(1, ch4, kernel_size=3)
        self.N = N
        self.kernel_dim = self.N * (ch4 * self.Fs * 2) + self.N * ch4
        self.F = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            conv(ch4, self.kernel_dim, kernel_size=1, act=None))

        # self.conv4_4=checkpoint_wrapper(self.conv4_4)
        # self.conv_res=checkpoint_wrapper(self.conv_res)
        # self.DME=checkpoint_wrapper(self.DME)
        # self.conv_DME=checkpoint_wrapper(self.conv_DME)
        # self.F=checkpoint_wrapper(self.F)
        self.init_F()

    def weights_init_F(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight, gain=self.wiF)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def init_F(self):
        self.F.apply(self.weights_init_F)

    ##########################################################################
    def forward(self, input_img, C):
        # C = torch.clone(input_img[:, :3, :, :])
        # feature extractor
        f1 = self.conv1_3(self.conv1_2(self.conv1_1(C)))
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(f1)))
        f3 = self.conv3_3(self.conv3_2(self.conv3_1(f2)))
        f_C = self.conv4_3(self.conv4_2(self.conv4_1(f3)))

        # filter encoder
        f = self.kconv1_3(self.kconv1_2(self.kconv1_1(input_img)))
        f = self.kconv2_3(self.kconv2_2(self.kconv2_1(f)))
        f = self.kconv3_3(self.kconv3_2(self.kconv3_1(f)))
        f = self.kconv4_3(self.kconv4_2(self.kconv4_1(f)))

        # disparity map estimator
        DM = self.DME(f)

        # filter predictor
        f_DM = self.conv_DME(DM)
        f = self.conv4_4(torch.cat([f, f_DM], 1))
        F = self.F(f)

        # IAC
        f = IAC(f_C, F, self.N, self.ch4, self.Fs)

        # reconstructor
        f = self.conv_res(f)

        f = self.upconv3_u(f) + f3
        f = self.upconv3_2(self.upconv3_1(f))

        f = self.upconv2_u(f) + f2
        f = self.upconv2_2(self.upconv2_1(f))

        f = self.upconv1_u(f) + f1
        f = self.upconv1_2(self.upconv1_1(f))

        out = self.out_res(f) + C

        # results
        # outs = collections.OrderedDict()

        # if is_train is False:
        #     outs['result'] = torch.clip(out, 0, 1.0)
        # else:
        #     outs['result'] = out
        #     # F
        #     outs['Filter'] = F
        #
        #     # DME
        #     f = self.kconv1_3(self.kconv1_2(self.kconv1_1(torch.cat([R, L], axis=1))))
        #     f = self.kconv2_3(self.kconv2_2(self.kconv2_1(f)))
        #     f = self.kconv3_3(self.kconv3_2(self.kconv3_1(f)))
        #     f = self.kconv4_3(self.kconv4_2(self.kconv4_1(f)))
        #     DM = self.DME(f)
        #     f_R_warped = DPD(Func.interpolate(R, scale_factor=1 / 8, mode='area'), DM, padding_mode='zeros',
        #                      device=self.device)
        #     outs['f_R_w'] = f_R_warped
        #     outs['f_L'] = Func.interpolate(L, scale_factor=1 / 8, mode='area')

        return out


if __name__ == '__main__':
    model = MYIR2()
    model.cuda()
    from torchsummary import summary
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # from thop import profile
    x = summary(model, (6, 256, 256))
    a = 1
    # flops, params = profile(model, inputs=(torch.randn(1,6,256,256).cuda(),))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
