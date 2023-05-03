import torch
from torch.utils import data as data
from torchvision import transforms
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_DP_paths_from_folder
from basicsr.data.transforms import paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, img2tensor
from basicsr.utils.img_util import imfrombytesDP, padding_DP
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR'], opt['dataroot_lqC']],
            ['lqL', 'lqR', 'gt', 'c'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

        # self.cache_data = {}

    def __getitem__(self, index):
        # if self.cache_data.get(index):
        #     return self.cache_data.get(index)
        # else:
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        img_paths = ['gt_path', 'lqL_path', 'lqR_path', 'c_path']
        client_keys = ['gt', 'lqL', 'lqR', 'c']
        imgs_np = []
        for path, client_key in zip(img_paths, client_keys):
            img_bytes = self.file_client.get(self.paths[index][path], client_key)
            try:
                imgs_np.append(imfrombytesDP(img_bytes, float32=True))
            except:
                raise Exception("gt path {} not working".format(path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            # padding
            img_lqL, img_lqR, img_gt, img_c = padding_DP(*imgs_np)

            # random crop
            img_lqL, img_lqR, img_gt, img_c = paired_random_crop_DP(img_lqL, img_lqR, img_gt, img_c, scale)

            # flip, rotation
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt, img_c = random_augmentation(img_lqL, img_lqR, img_gt, img_c)
        else:
            img_lqL, img_lqR, img_gt, img_c = imgs_np

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt, img_c = img2tensor([img_lqL, img_lqR, img_gt, img_c], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_c, self.mean, self.std, inplace=True)

        # resize = transforms.Resize([128, 128], antialias=True)
        # img_lqL = resize(img_lqL)
        # img_lqR = resize(img_lqR)
        # img_gt = resize(img_gt)
        try:
            img_lq = torch.cat([img_lqR, img_lqL], 0)
        except:
            a = 1
        # self.cache_data[index] = {
        #     'lq': img_lq,
        #     # 'lq': img_lqR,
        #     'gt': img_gt,
        #     'lq_path': lqL_path,
        #     'gt_path': gt_path
        # }
        return {
            'lq': img_lq,
            # 'lq': img_lqR,
            'gt': img_gt,
            'c': img_c,
            'lq_path': self.paths[index]['lqL_path'],
            'gt_path': self.paths[index]['gt_path']
        }

    def __len__(self):
        return len(self.paths)
