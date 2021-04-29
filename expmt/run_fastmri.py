#!/usr/bin/env python

# TODO's
# - update script functionality per qdess, e.g. command line tools
# - check scaling factor of ksp_masked

import os, sys
import numpy as np
import torch

from utils.data_io import get_mask
from include.decoder_conv import init_convdecoder
from include.fit import fit
from utils.transform import fft_2d, ifft_2d, root_sum_squares, \
                            reshape_adj_channels_to_complex_vals, \
                            crop_center

dtype = torch.cuda.FloatTensor
path_out = '/bmrNAS/people/dvv/out_fastmri/'
dim = 320

def run_expmt(file_id_list):

    for file_id in file_id_list:

        if os.path.exists('{}{}_dc.npy'.format(path_out, file_id)):
            continue

        f, ksp_orig = load_h5_fastmri(file_id)
        ksp_orig = torch.from_numpy(ksp_orig)

        mask = get_mask(ksp_orig)

        net, net_input, ksp_orig_ = init_convdecoder(ksp_orig, mask)

        ksp_masked = 0.1 * ksp_orig_ * mask 

        net = fit(ksp_masked, net, net_input, mask)

        img_out = net(net_input.type(dtype))
        img_out = reshape_adj_channels_to_complex_vals(img_out[0])
        ksp_est = fft_2d(img_out)
        ksp_dc = torch.where(mask, ksp_masked, ksp_est)

        #img_est = crop_center(root_sum_squares(ifft_2d(ksp_est)).detach(), dim, dim)
        img_dc = crop_center(root_sum_squares(ifft_2d(ksp_dc)).detach(), dim, dim)
        img_gt = crop_center(root_sum_squares(ifft_2d(ksp_orig)), dim, dim)
        # note: use unscaled ksp_orig to make gt -- different from old fastmri processing

        np.save('{}{}_dc.npy'.format(path_out, file_id), img_dc)
        np.save('{}{}_gt.npy'.format(path_out, file_id), img_gt)


if __name__ == '__main__':

    test_set = ['1000000', '1000007',
                '1000017', '1000026', '1000031', '1000033', '1000041', '1000052',
                '1000071', '1000073', '1000107', '1000108', '1000114', '1000126',
                '1000153', '1000178', '1000182', '1000190', '1000196', '1000201',
                '1000206', '1000229', '1000243', '1000247', '1000254', '1000263',
                '1000264', '1000267', '1000273', '1000277', '1000280', '1000283', 
                '1000291', '1000292', '1000308', '1000325', '1000464', '1000537']

    run_expmt(test_set)
