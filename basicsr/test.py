
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

from basicsr.archs.my_arch import MYIR

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
def MAE(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])

def PSNR(img1, img2):
    mse_ = np.mean( (img1 - img2) ** 2 )
    if mse_ == 0:
        return 100
    return 10 * math.log10(1 / mse_)

def SSIM(img1, img2):
    return metrics.structural_similarity(img1, img2, data_range=1, multichannel=True,channel_axis=-1)

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def load_img16(filepath):
    return cv2.cvtColor(cv2.imread(filepath, -1), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


alex = lpips.LPIPS(net='alex').cuda()

parser = argparse.ArgumentParser(description='Dual Pixel Defocus Deblurring using Restormer')

parser.add_argument('--input_dir', default='/root/autodl-tmp/test/', type=str, help='Directory of validation images')
# parser.add_argument('--input_dir', default='/home/lab/code1/Restomer/Defocus_Deblurring/Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/root/autodl-tmp/pycharm_project_983/results/Dual_Pixel_Defocus_Deblurring/', type=str,
                    help='Directory for results')
parser.add_argument('--weights',
                    # default='/home/lab/code1/IR/experiments/train_MYIR_scratch/models/net_g_160000.pth', type=str,
                    default='/root/autodl-tmp/pycharm_project_983/experiments/train_MYIR_scratch/models/net_g_138000.pth', type=str,
                    help='Path to weights')
parser.add_argument('--save_images', default=True, help='Save denoised images in result directory')

args = parser.parse_args()

####### Load yaml #######
# yaml_file = '/home/lab/code1/IR/options/train/SwinIR/train_MYIR_scratch.yml'
yaml_file = '/root/autodl-tmp/pycharm_project_983/options/train/SwinIR/train_MYIR_scratch3.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################
device = torch.device("cuda")
model_restoration = MYIR(**x['network_g'])
device_id = torch.cuda.current_device()

checkpoint = torch.load(args.weights, map_location=lambda storage, loc: storage.cuda(device_id))
model_restoration.load_state_dict(checkpoint['params'])

print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

result_dir = args.result_dir
if args.save_images:
    os.makedirs(result_dir, exist_ok=True)

filesL = natsorted(glob(os.path.join(args.input_dir, 'inputL', '*.png')))
filesR = natsorted(glob(os.path.join(args.input_dir, 'inputR', '*.png')))
filesC = natsorted(glob(os.path.join(args.input_dir, 'target', '*.png')))


indoor_labels = np.load(args.input_dir+'indoor_labels.npy')
outdoor_labels = np.load(args.input_dir+'outdoor_labels.npy')

psnr, mae, ssim, pips = [], [], [], []
with torch.no_grad():
    for fileL, fileR, fileC in tqdm(zip(filesL, filesR, filesC), total=len(filesC)):

        # imgL = np.float32(load_img16(fileL)) / 65535.
        # imgR = np.float32(load_img16(fileR)) / 65535.
        # imgC = np.float32(load_img16(fileC)) / 65535.
        # imgL = torch.from_numpy(imgL).permute(2,0,1)
        # imgR = torch.from_numpy(imgR).permute(2,0,1)
        # imgC = torch.from_numpy(imgC).permute(2,0,1)
        # resize = transforms.Resize([512, 512], antialias=True)
        # imgC = resize(imgC).numpy()
        # imgR = resize(imgR).numpy()
        # imgL = resize(imgL).numpy()
        # patchC = torch.from_numpy(imgC).unsqueeze(0).cuda()
        # patchL = torch.from_numpy(imgL).unsqueeze(0)
        # patchR = torch.from_numpy(imgR).unsqueeze(0)
        # input_ = torch.cat([patchL, patchR], 1)
        # restored = model_restoration(input_.cuda())
        # # restored = patchR.cuda()
        # restored = torch.clamp(restored, 0, 1)
        # pips.append(alex(patchC, restored, normalize=True).item())

        imgL = np.float32(load_img16(fileL)) / 65535.
        imgR = np.float32(load_img16(fileR)) / 65535.
        imgC = np.float32(load_img16(fileC)) / 65535.
        patchC = torch.from_numpy(imgC).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        patchL = torch.from_numpy(imgL).unsqueeze(0).permute(0, 3, 1, 2)
        patchR = torch.from_numpy(imgR).unsqueeze(0).permute(0, 3, 1, 2)
        input_ = torch.cat([patchL, patchR], 1)
        # restored = imgR
        restored = model_restoration(input_)
        restored = torch.clamp(restored, 0, 1)
        pips.append(alex(patchC, restored, normalize=True).item())


        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        imgC = torch.from_numpy(imgC).permute(1,2,0).numpy()
        psnr.append(PSNR(imgC, restored))
        mae.append(MAE(imgC, restored))
        ssim.append(SSIM(imgC, restored))
        print(PSNR(imgC, restored))
        print(MAE(imgC, restored))
        print(SSIM(imgC, restored))
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

# Overall: PSNR 27.515790 SSIM 0.863139 MAE 0.033217 LPIPS 0.863139
# Indoor:  PSNR 30.421379 SSIM 0.924711 MAE 0.021649 LPIPS 0.021649
# Outdoor: PSNR 24.759205 SSIM 0.804724 MAE 0.044192 LPIPS 0.021649

# Overall: PSNR 25.979450 SSIM 0.807825 MAE 0.039890 LPIPS 0.807825
# Indoor:  PSNR 28.717393 SSIM 0.885318 MAE 0.028145 LPIPS 0.028145
# Outdoor: PSNR 23.381916 SSIM 0.734307 MAE 0.051032 LPIPS 0.028145

# Overall: PSNR 25.444740 SSIM 0.783585 MAE 0.040402 LPIPS 0.783585
# Indoor:  PSNR 28.132499 SSIM 0.859354 MAE 0.026925 LPIPS 0.026925
# Outdoor: PSNR 22.894816 SSIM 0.711701 MAE 0.053187 LPIPS 0.026925