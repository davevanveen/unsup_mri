#!/usr/bin/env python

import os, sys
import numpy as np
import torch
import argparse

from include.decoder_conv import init_convdecoder
from include.fit import fit_many_heads as fit
from include.mri_helpers import apply_mask
from utils.data_io import load_qdess
from utils.transform import fft_2d, ifft_2d, root_sum_squares, \
                            reshape_adj_channels_to_complex_vals

dtype = torch.cuda.FloatTensor

TEST_SET = ['005', '006', '030', '034', '048', '052', '065', '066', '080', 
            '096', '099', '120']
ACCEL_LIST = [4, 8] 


def run_expmt(args):

    for file_id in args.file_id_list: 

        ksp_orig = load_qdess(file_id, idx_kx=None) # default central slice in kx (axial)

        for accel in args.accel_list:

            # manage paths for input/output
            path_base = '/bmrNAS/people/dvv/out_qdess/accel_{}x/'.format(accel)
            path_out = '{}{}/'.format(path_base, args.dir_out)
            args.path_gt = path_base + 'gt/'
            if os.path.exists('{}MTR_{}_e1.npy'.format(path_out, file_id)):
                continue
            if not os.path.exists(path_out):
                os.makedirs(path_out)
            if not os.path.exists(args.path_gt):
                os.makedirs(args.path_gt)

            # initialize network 
            net1, net_input1, ksp_orig_ = init_convdecoder(ksp_orig, fix_random_seed=False) 
            net2, net_input2, _ = init_convdecoder(ksp_orig, fix_random_seed=False) 

            # apply mask after rescaling k-space. want complex tensors dim (nc, ky, kz)
            ksp_masked, mask = apply_mask(ksp_orig_, accel, calib=args.calib, expmt=True)
            
            # fit network, get net output - default 10k iterations, lam_tv=1e-8
            net1, net2 = fit(ksp_masked=ksp_masked, net1=net1, net_input1=net_input1, 
                                                    net2=net2, net_input2=net_input2, mask=mask)
            im_out1 = net1(net_input1.type(dtype)) # real tensor dim (2*nc, kx, ky)
            im_out2 = net2(net_input2.type(dtype)) # real tensor dim (2*nc, kx, ky)
            im_out = torch.mean(torch.stack([im_out1, im_out2]), dim=0)
            im_out = reshape_adj_channels_to_complex_vals(im_out[0]) # complex tensor dim (nc, kx, ky)
            
            # perform dc step
            ksp_est = fft_2d(im_out)
            ksp_dc = torch.where(mask, ksp_masked, ksp_est)
            np.save('{}/MTR_{}_ksp_dc.npy'.format(path_out, file_id), ksp_dc.detach().numpy())

            # create data-consistent, ground-truth images from k-space
            im_1_dc = root_sum_squares(ifft_2d(ksp_dc[:8])).detach()
            im_2_dc = root_sum_squares(ifft_2d(ksp_dc[8:])).detach()
            np.save('{}MTR_{}_e1.npy'.format(path_out, file_id), im_1_dc)
            np.save('{}MTR_{}_e2.npy'.format(path_out, file_id), im_2_dc)

            # save gt w proper array scaling if dne
            if not os.path.exists('{}MTR_{}_e1_gt.npy'.format(args.path_gt, file_id)):
                im_1_gt = root_sum_squares(ifft_2d(ksp_orig[:8]))
                im_2_gt = root_sum_squares(ifft_2d(ksp_orig[8:]))
                np.save('{}MTR_{}_e1_gt.npy'.format(args.path_gt, file_id), im_1_gt)
                np.save('{}MTR_{}_e2_gt.npy'.format(args.path_gt, file_id), im_2_gt)
            
            print('recon {}'.format(file_id)) 

    return

def init_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--accel_list', nargs='+', type=int, default=ACCEL_LIST)
    parser.add_argument('--file_id_list', nargs='+', default=TEST_SET)
    parser.add_argument('--dir_out', type=str, default='')
    parser.add_argument('--calib', type=int, default=64)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = init_parser()

    run_expmt(args)
