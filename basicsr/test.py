import numpy as np
import os
import argparse
from tqdm import tqdm
import math
from torchvision import transforms
import torch.nn as nn
import torch
import cv2
from skimage import metrics
from sklearn.metrics import mean_absolute_error
from natsort import natsorted
from glob import glob

import lpips

from basicsr.archs.my2_arch import MYIR2

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def MAE(img1, img2):
    mae_0 = mean_absolute_error(img1[:, :, 0], img2[:, :, 0],
                                multioutput='uniform_average')
    mae_1 = mean_absolute_error(img1[:, :, 1], img2[:, :, 1],
                                multioutput='uniform_average')
    mae_2 = mean_absolute_error(img1[:, :, 2], img2[:, :, 2],
                                multioutput='uniform_average')
    return np.mean([mae_0, mae_1, mae_2])


def PSNR(img1, img2):
    mse_ = np.mean((img1 - img2) ** 2)
    if mse_ == 0:
        return 100
    return 10 * math.log10(1 / mse_)


def SSIM(img1, img2):
    return metrics.structural_similarity(img1, img2, data_range=1, multichannel=True, channel_axis=-1)


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def load_img16(filepath):
    return cv2.cvtColor(cv2.imread(filepath, -1), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# alex = lpips.LPIPS(net='alex').cuda()

parser = argparse.ArgumentParser(description='Dual Pixel Defocus Deblurring using Restormer')

# parser.add_argument('--input_dir', default='/root/autodl-tmp/test/', type=str, help='Directory of validation images')
parser.add_argument('--input_dir', default='/home/lab/code1/Restomer/Defocus_Deblurring/Datasets/test/', type=str,
                    help='Directory of validation images')
parser.add_argument('--result_dir',
                    default='/root/autodl-tmp/pycharm_project_983/results/Dual_Pixel_Defocus_Deblurring/', type=str,
                    help='Directory for results')
parser.add_argument('--weights',
                    # default='/home/lab/code1/IR/experiments/train_MYIR_scratch/models/net_g_74000.pth', type=str,
                    default='/data/code/IFAN/ckpt/IFAN_dual.pytorch', type=str,
                    # default='/root/autodl-tmp/pycharm_project_983/experiments/train_MYIR_scratch/models/net_g_144000.pth', type=str,
                    help='Path to weights')
parser.add_argument('--save_images', default=True, help='Save denoised images in result directory')

args = parser.parse_args()

####### Load yaml #######
yaml_file = '/home/lab/code1/IR/options/train/SwinIR/train_MYIR_scratch2.yml'
# yaml_file = '/root/autodl-tmp/pycharm_project_983/options/train/SwinIR/train_MYIR_scratch3.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################
device = torch.device("cuda")
model_restoration = MYIR2(**x['network_g'])
device_id = torch.cuda.current_device()

checkpoint = torch.load(args.weights, map_location=lambda storage, loc: storage.cuda(device_id))
# checkpoint = torch.load(args.weights,)
a = {}
for key, v in checkpoint.items():
    if not key[15:].startswith('t.RBF'):
        a[key[15:]] = v
# model_restoration.load_state_dict(checkpoint['params'])
model_restoration.load_state_dict(a)

print("===>Testing using weights: ", args.weights)
# model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

result_dir = args.result_dir
if args.save_images:
    os.makedirs(result_dir, exist_ok=True)

filesL = natsorted(glob(os.path.join(args.input_dir, 'inputL', '*.png')))
filesR = natsorted(glob(os.path.join(args.input_dir, 'inputR', '*.png')))
filesC = natsorted(glob(os.path.join(args.input_dir, 'target', '*.png')))

indoor_labels = np.load(args.input_dir + 'indoor_labels.npy')
outdoor_labels = np.load(args.input_dir + 'outdoor_labels.npy')


# ---------------------
def read_frame(path, norm_val=None, rotate=None):
    if norm_val == (2 ** 16 - 1):
        frame = cv2.imread(path, -1)
        if rotate is not None:
            frame = cv2.rotate(frame, rotate)
        frame = frame / norm_val
        frame = frame[..., ::-1]
    else:
        frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if rotate is not None:
            frame = cv2.rotate(frame, rotate)
        frame = frame / 255.
    return np.expand_dims(frame, axis=0)


def refine_image(img, val=8):
    shape = img.shape
    if len(shape) == 4:
        _, h, w, _ = shape[:]
        return img[:, 0: h - h % val, 0: w - w % val, :]
    elif len(shape) == 3:
        h, w = shape[:2]
        return img[0: h - h % val, 0: w - w % val, :]
    elif len(shape) == 2:
        h, w = shape[:2]
        return img[0: h - h % val, 0: w - w % val]


# ---------------------

psnr, mae, ssim, pips = [], [], [], []
with torch.no_grad():
    for fileL, fileR, fileC in tqdm(zip(filesL, filesR, filesC), total=len(filesC)):

        # imgL = np.float32(load_img16(fileL)) / 65535.
        # imgR = np.float32(load_img16(fileR)) / 65535.
        # imgC = np.float32(load_img16(fileC)) / 65535.
        # imgCC = np.float32(load_img16('/data/junyonglee/defocus_deblur/DPDD/test_c/source/' + fileC[-12:])) / 65535.
        # patchCC = torch.from_numpy(imgCC).unsqueeze(0).permute(0, 3, 1, 2).to('cuda')
        # patchC = torch.from_numpy(imgC).unsqueeze(0).permute(0, 3, 1, 2).to('cuda')
        # patchL = torch.from_numpy(imgL).unsqueeze(0).permute(0, 3, 1, 2)
        # patchR = torch.from_numpy(imgR).unsqueeze(0).permute(0, 3, 1, 2)
        # input_ = torch.cat([patchL, patchR], 1)

        imgL = refine_image(read_frame(fileL, 255, None), 8)
        imgR = refine_image(read_frame(fileR, 255, None), 8)
        imgC = refine_image(read_frame(fileC, 255, None), 8)
        imgCC = refine_image(read_frame('/data/junyonglee/defocus_deblur/DPDD/test_c/source/'+fileC[-12:], 255, None), 8)
        patchC = torch.FloatTensor(imgC.transpose(0, 3, 1, 2).copy()).to('cuda')
        patchL = torch.FloatTensor(imgL.transpose(0, 3, 1, 2).copy()).to('cuda')
        patchR = torch.FloatTensor(imgR.transpose(0, 3, 1, 2).copy()).to('cuda')
        patchCC = torch.FloatTensor(imgCC.transpose(0, 3, 1, 2).copy()).to('cuda')
        input_ = torch.cat([patchR, patchL], 1)

        # restored = imgR

        #  if split
        # input_1 = torch.clone(input_[:, :, :, :input_.shape[3] // 2])
        # input_2 = torch.clone(input_[:, :, :, input_.shape[3] // 2:])
        # input_1C = torch.clone(patchCC[:, :, :, :input_.shape[3] // 2])
        # input_2C = torch.clone(patchCC[:, :, :, input_.shape[3] // 2:])
        # input_1 = model_restoration(input_1.cuda(0), input_1C)
        # input_2 = model_restoration(input_2.cuda(0), input_2C)
        # restored = torch.cat([input_1, input_2], 3)

        # else:
        restored = model_restoration(input_.cuda(0), patchCC.cuda(0))

        #
        restored = torch.clamp(restored, 0, 1)
        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()


        # caculate
        # psnr.append(PSNR(imgC, restored))
        # print(PSNR(imgC, restored))
        # mae.append(MAE(imgC, restored))
        # ssim.append(SSIM(imgC, restored))

        gt = patchC.cpu().numpy()[0].transpose(1, 2, 0)
        psnr.append(PSNR(gt, restored))
        print(PSNR(gt, restored))
        mae.append(MAE(gt, restored))
        ssim.append(SSIM(gt, restored))


        # pips.append(alex(patchC, restored, normalize=True).item())
        # if args.save_images:
        #     save_file = os.path.join(result_dir, os.path.split(fileC)[-1])
        #     restored = np.uint16((restored * 65535).round())
        #     utils.save_img(save_file, restored)

psnr, mae, ssim, pips = np.array(psnr), np.array(mae), np.array(ssim), np.array(ssim)

psnr_indoor, mae_indoor, ssim_indoor, pips_indoor = psnr[indoor_labels - 1], mae[indoor_labels - 1], ssim[
    indoor_labels - 1], ssim[indoor_labels - 1]
psnr_outdoor, mae_outdoor, ssim_outdoor, pips_outdoor = psnr[outdoor_labels - 1], mae[outdoor_labels - 1], ssim[
    outdoor_labels - 1], ssim[outdoor_labels - 1]

print("Overall: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr), np.mean(ssim), np.mean(mae),
                                                                    np.mean(pips)))
print("Indoor:  PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_indoor), np.mean(ssim_indoor),
                                                                    np.mean(mae_indoor), np.mean(mae_indoor)))
print("Outdoor: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_outdoor), np.mean(ssim_outdoor),
                                                                    np.mean(mae_outdoor), np.mean(mae_indoor)))
# Overall: PSNR 23.785344 SSIM 0.716311 MAE 0.047779 LPIPS 0.716311
# Indoor:  PSNR 26.217412 SSIM 0.795828 MAE 0.034156 LPIPS 0.034156
# Outdoor: PSNR 21.477998 SSIM 0.640872 MAE 0.060703 LPIPS 0.034156

# Overall: PSNR 25.566490 SSIM 0.794726 MAE 0.038873 LPIPS 0.794726
# Indoor:  PSNR 28.258082 SSIM 0.865417 MAE 0.025477 LPIPS 0.025477
# Outdoor: PSNR 23.012930 SSIM 0.727660 MAE 0.051581 LPIPS 0.025477

# Overall: PSNR 24.534078 SSIM 0.756053 MAE 0.043077 LPIPS 0.756053
# Indoor:  PSNR 27.026428 SSIM 0.831138 MAE 0.029181 LPIPS 0.029181
# Outdoor: PSNR 22.169542 SSIM 0.684820 MAE 0.056260 LPIPS 0.029181
# Overall: PSNR 24.497103 SSIM 0.753728 MAE 0.043261 LPIPS 0.753728
# Indoor:  PSNR 26.988409 SSIM 0.828767 MAE 0.029318 LPIPS 0.029318
# Outdoor: PSNR 22.133556 SSIM 0.682537 MAE 0.056488 LPIPS 0.029318
# Overall: PSNR 26.023693 SSIM 0.805778 MAE 0.037130 LPIPS 0.805778
# Indoor:  PSNR 28.699427 SSIM 0.870385 MAE 0.024749 LPIPS 0.024749
# Outdoor: PSNR 23.485176 SSIM 0.744484 MAE 0.048877 LPIPS 0.024749