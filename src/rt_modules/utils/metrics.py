# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from scipy import signal
from scipy import ndimage
from pytorch_msssim import ms_ssim as MS_SSIM
import torch.nn as nn
import torch
def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def calc_ssim(img1, img2, data_range=255):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2

    return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                         (sigma1_sq + sigma2_sq + C2)),
            (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))


def calc_msssim(img1, img2, data_range=255):
    '''
    img1 and img2 are 2D arrays
    '''
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    height, width = img1.shape
    if height < 176 or width < 176:
        # according to HM implementation
        level = 4
        weight = np.array([0.0517, 0.3295, 0.3462, 0.2726])
    if height < 88 or width < 88:
        assert False
    downsample_filter = np.ones((2, 2)) / 4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(level):
        ssim_map, cs_map = calc_ssim(im1, im2, data_range=data_range)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter,
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter,
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level - 1]**weight[0:level - 1]) *
            (mssim[level - 1]**weight[level - 1]))


def calc_msssim_rgb(img1, img2, data_range=255):
    '''
    img1 and img2 are arrays with 3xHxW
    '''
    msssim = 0
    for i in range(3):
        msssim += calc_msssim(img1[i, :, :], img2[i, :, :], data_range)
    return msssim / 3


def calc_psnr(img1, img2, data_range=255):
    '''
    img1 and img2 are arrays with same shape
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean(np.square(img1 - img2))
    if np.isnan(mse) or np.isinf(mse):
        return -999.9
    if mse > 1e-10:
        psnr = 10 * np.log10(data_range * data_range / mse)
    else:
        psnr = 999.9
    if psnr > 99.9:
        psnr = 99.9
    return psnr

def msssim_loss(fake, real):
    assert fake.shape == real.shape
    return 1 - MS_SSIM(fake, real, data_range=1.0, size_average=False)



def color_conv_matrix(type = "709"):
    if type=="601":
        # BT.601
        a = 0.299
        b = 0.587
        c = 0.114
        d = 1.772
        e = 1.402
    elif type=="709":
        # BT.709
        a = 0.2126
        b = 0.7152
        c = 0.0722
        d = 1.8556
        e = 1.5748
    elif type=="2020":
        # BT.2020
        a = 0.2627
        b = 0.6780
        c = 0.0593
        d = 1.8814
        e = 1.4747
    else:
        raise NotImplementedError

    return a,b,c,d,e



    
class MSSSIMLossYUV2YUV(nn.Module):
    r"""Input RGB, calculate MS-SSIM in YUV format.
    """
    def __init__(self, gamma=8):
        super(MSSSIMLossYUV2YUV, self).__init__()
        self.gamma = gamma

    def forward(self, img1, img2):
        r"""
        Args: 
            img1: First RGB Image.
            img2: Second RGB Image.

        Returns:
            distortion_loss : Loss of MS-SSIM in YUV
            Y_distortion    : MS-SSIM in Y
            distortion      : MS-SSIM in YUV
        """
        # rgb to yuv444
        img_yuv1 = torch.clamp(img1, 0, 1)
        img_yuv2 = torch.clamp(img2, 0, 1)
        y1, u1, v1 = img_yuv1.split([1,1,1], dim=1)
        y2, u2, v2 = img_yuv2.split([1,1,1], dim=1)
        # yuv444 to yuv420
        u1 = u1[:,:,::2,::2]
        u2 = u2[:,:,::2,::2]
        v1 = v1[:,:,::2,::2]
        v2 = v2[:,:,::2,::2]
        # calculate loss
        Y_loss = 1 - MS_SSIM(y1, y2, data_range=1.0, size_average=False)
        U_loss = 1 - MS_SSIM(u1, u2, data_range=1.0, size_average=False,win_size=7)
        V_loss = 1 - MS_SSIM(v1, v2, data_range=1.0, size_average=False,win_size=7)
        distortion_loss =  (self.gamma * Y_loss + U_loss + V_loss) / (self.gamma + 2)
        return distortion_loss#, Y_distortion, U_distortion, V_distortion



    
def charb(x, alpha, eps):
    '''charbonnier Loss function'''
    return torch.mean(torch.pow(x.pow(2) + eps, 1. / alpha))

def charbnomean(x, alpha, eps):
    '''charbonnier Loss function'''
    return torch.pow(x.pow(2) + eps, 1. / alpha)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, alpha=2,eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, input, target):
        return charb(input - target, self.alpha, self.eps)

class CharbonniernomeanLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, alpha=2,eps=1e-6):
        super(CharbonniernomeanLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, input, target):
        return charbnomean(input - target, self.alpha, self.eps)
