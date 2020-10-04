import os, sys
import numpy as np
import torch

from utils.data_io import load_h5
from utils.transform import recon_ksp_to_img
from utils.helpers import get_masks
from include.decoder_conv import init_convdecoder
from include.mri_helpers import get_masked_measurements, data_consistency
from include.fit import fit
from utils.evaluate import calc_metrics

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    torch.cuda.set_device(2)
else:
    dtype = torch.FloatTensor


def run():
    
    # .h5 files in hard-coded path /bmrNAS/people/dvv/multicoil_val/
    FILE_ID = '1000464' 
    NUM_ITER = 10000

    f, slice_ksp = load_h5(FILE_ID) # load full mri measurements

    mask, mask2d, mask1d = get_masks(f, slice_ksp) 

    # initialize net, net input seed, and scale slice_ksp
    net, net_input, slice_ksp = init_convdecoder(slice_ksp, mask)
    img_gt = recon_ksp_to_img(slice_ksp) # must do this after scaling slice_ksp

    # apply mask to measurements for fitting model
    ksp_masked, img_masked = get_masked_measurements(slice_ksp, mask)

    net, mse_wrt_ksp, mse_wrt_img = fit(
        ksp_masked=ksp_masked, img_masked=img_masked,
        net=net, net_input=net_input, mask2d=mask2d,
        img_ls=None, num_iter=NUM_ITER, dtype=dtype)

    img_out = net(net_input.type(dtype))[0] # estimate image \hat{x} = G(\hat{C})

    img_dc, img_est = data_consistency(img_out, slice_ksp, mask1d)

    _, _, ssim_, psnr_ = calc_metrics(img_dc, img_gt)


if __name__ == '__main__':
    run()
