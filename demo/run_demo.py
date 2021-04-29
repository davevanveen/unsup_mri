#!/usr/bin/env python

''' run simple demo of unsup mri recon using a fastmri sample '''

import os, sys
import numpy as np
import torch

from include.decoder_conv import init_convdecoder
from include.fit import fit
from utils.data_io import load_h5_fastmri, get_mask
from utils.transform import fft_2d, ifft_2d, root_sum_squares, \
                            reshape_adj_channels_to_complex_vals, \
                            crop_center

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else: 
    torch.FloatTensor
dim = 320

def run_demo():

    ksp_orig = load_h5_fastmri(file_id=None, demo=True)
    
    mask = get_mask(ksp_orig)

    net, net_input, ksp_orig_ = init_convdecoder(ksp_orig)

    ksp_masked = 0.1 * ksp_orig_ * mask 

    net = fit(ksp_masked, net, net_input, mask)

    img_out = net(net_input.type(dtype))
    img_out = reshape_adj_channels_to_complex_vals(img_out[0])
    ksp_est = fft_2d(img_out)
    ksp_dc = torch.where(mask, ksp_masked, ksp_est)

    img_dc = crop_center(root_sum_squares(ifft_2d(ksp_dc)).detach(), dim, dim)
    img_gt = crop_center(root_sum_squares(ifft_2d(ksp_orig)), dim, dim)
    img_zf = crop_center(root_sum_squares(ifft_2d(ksp_masked)), dim, dim)

    np.save('data/out.npy', img_dc)
    np.save('data/gt.npy', img_gt)
    np.save('data/zf.npy', img_zf) 


if __name__ == '__main__':

    run_demo()
