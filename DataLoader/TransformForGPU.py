import numpy as np
import torch
import torch.nn.functional as F

from .Utils import randomTransform


class RandomAffineTransform(object):
    def __init__(self,
                 rotation_degree=[-45, 45],
                 translate_interval=[-10, 10],
                 scale_interval=[0.8, 1.2],
                 image_size=[200, 200]):
        self.rotation_degree = rotation_degree
        self.translate_interval = translate_interval
        self.scale_interval = scale_interval
        self.image_size = image_size

    def getAffineMatrix(self, batch_size):
        affine_matrix_list = []
        for i in range(batch_size):
            affine_matrix_list.append(
                randomTransform(self.rotation_degree, self.translate_interval,
                                self.scale_interval, self.image_size))
        return torch.tensor(np.array(affine_matrix_list), dtype=torch.float)

    def __call__(self, pair):
        batch_size = pair['src']['img'].size()[0]
        theta = self.getAffineMatrix(batch_size).cuda()
        phi = F.affine_grid(theta,
                            pair['src']['img'].size(),
                            align_corners=True)
        pair['src']['img'] = F.grid_sample(pair['src']['img'],
                                           phi,
                                           mode='bilinear',
                                           align_corners=True)
        pair['tgt']['img'] = F.grid_sample(pair['tgt']['img'],
                                           phi,
                                           mode='bilinear',
                                           align_corners=True)
        return pair


class CentralCropTensor(object):
    def __init__(self, src_img_size, tgt_img_size):
        self.center = (src_img_size[1] // 2, src_img_size[0] // 2)
        self.halfheight = tgt_img_size[0] // 2
        self.halfwidth = tgt_img_size[1] // 2

    def crop(self, img):
        cropped_img = img[:, :, self.center[1] -
                          self.halfheight:self.center[1] + self.halfheight,
                          self.center[0] - self.halfwidth:self.center[0] +
                          self.halfwidth]
        return cropped_img

    def __call__(self, pair):
        pair['src']['img'] = self.crop(pair['src']['img'])
        pair['tgt']['img'] = self.crop(pair['tgt']['img'])
        pair['src']['seg'] = self.crop(pair['src']['seg'])
        pair['tgt']['seg'] = self.crop(pair['tgt']['seg'])
        return pair


class NormalizeTensor(object):
    def normalize(self, tensor):
        return (tensor - torch.min(tensor, dim=0)[0]) / (
            torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0])

    def __call__(self, pair):
        pair['src']['img'] = self.normalize(pair['src']['img'])
        pair['tgt']['img'] = self.normalize(pair['tgt']['img'])
        return pair


class RandomMirrorTensor2D(object):
    @staticmethod
    def getParams():
        xflip = np.random.choice([0, 1])
        yflip = np.random.choice([0, 1])
        return xflip, yflip

    @staticmethod
    def flip(img: torch.Tensor, xflip, yflip) -> torch.Tensor:
        if not xflip and not yflip:
            return img
        dims = []
        if xflip:
            dims.append(2)
        if yflip:
            dims.append(3)
        return torch.flip(img, dims)

    def __call__(self, pair):
        xflip, yflip = self.getParams()
        pair['src']['img'] = self.flip(pair['src']['img'], xflip, yflip)
        pair['tgt']['img'] = self.flip(pair['tgt']['img'], xflip, yflip)
        return pair
