import os, sys
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import h5py
import sigpy
from sigpy.mri.samp import poisson
import torch
import argparse

sys.path.append('/home/vanveen/ConvDecoder/')
from utils.data_io import load_h5_qdess, num_params
from include.decoder_conv import init_convdecoder
from include.fit import fit
from include.mri_helpers import apply_mask, apply_dual_mask
from utils.evaluate import calc_metrics
from utils.transform import fft_2d, ifft_2d, root_sum_squares, \
                            reshape_complex_vals_to_adj_channels, \
                            reshape_adj_channels_to_complex_vals

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor

TEST_SET = ['005', '006', '030', '034', '048', '052', '065', '066', '080', 
            '096', '099', '120']#, '144', '156', '158', '173', '176', '178', 
            #'188', '196', '198', '199', '218', '219', '221', '223',
            #'224', '227', '235', '237', '240', '241', '244', '248']
ACCEL_LIST = [4, 8] # 4, 6, 8]


def run_expmt(args):

    for file_id in args.file_id_list: 

        ksp = load_h5_qdess(file_id)

        # load, concat both echo slices
        idx_kx = ksp.shape[0] // 2 # want central slice in kx (axial) b/c we undersample in (ky,kz)
        ksp_echo1 = ksp[:,:,:,0,:].permute(3,0,1,2)[:, idx_kx, :, :]
        ksp_echo2 = ksp[:,:,:,1,:].permute(3,0,1,2)[:, idx_kx, :, :]
        ksp_orig = torch.cat((ksp_echo1, ksp_echo2), 0)

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
            net, net_input, ksp_orig_ = init_convdecoder(ksp_orig) # TODO: init dd+ w ksp_masked

            # apply mask after rescaling k-space. want complex tensors dim (nc, ky, kz)
            if args.dual_mask:
                ksp_masked, mask1, mask2 = apply_dual_mask(ksp_orig_, accel)
            else:
                ksp_masked, mask = apply_mask(ksp_orig_, accel)
            im_masked = ifft_2d(ksp_masked)

            # fit network, get net output - default 10k iterations, lam_tv=1e-8
            if args.dual_mask:
                net = fit(ksp_masked=ksp_masked, img_masked=im_masked,
                          net=net, net_input=net_input, mask=mask1, mask2=mask2)
            else: 
                net = fit(ksp_masked=ksp_masked, img_masked=im_masked,
                          net=net, net_input=net_input, mask=mask)
            im_out = net(net_input.type(dtype)) # real tensor dim (2*nc, kx, ky)
            im_out = reshape_adj_channels_to_complex_vals(im_out[0]) # complex tensor dim (nc, kx, ky)
            
            # perform dc step
            ksp_est = fft_2d(im_out)
            if args.dual_mask: #  ksp in format [re+im(e1), re+im(e2)]
                ksp_dc_e1 = torch.where(mask1, ksp_masked[:8], ksp_est[:8])
                ksp_dc_e2 = torch.where(mask2, ksp_masked[8:], ksp_est[8:])
                ksp_dc = torch.cat((ksp_dc_e1, ksp_dc_e2), 0)
            else:
                ksp_dc = torch.where(mask, ksp_masked, ksp_est)

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
    
    parser.add_argument('--dual_mask', dest='dual_mask', action='store_true')
    parser.add_argument('--no_dual_mask', dest='dual_mask', action='store_false')
    parser.set_defaults(no_dual_mask=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = init_parser()

    torch.cuda.set_device(args.gpu_id)

    run_expmt(args)
