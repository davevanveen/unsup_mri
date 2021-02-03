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
from utils.evaluate import calc_metrics
from utils.transform import fft_2d, ifft_2d, root_sum_squares, \
                            reshape_complex_vals_to_adj_channels, \
                            reshape_adj_channels_to_complex_vals

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor

TEST_SET = ['030', '034', '048', '052', '065', '066', '005', '006', '080', 
            '096', '099', '120', '144', '156', '158', '173', '176', '178', 
            '188', '196', '198', '199', '218', '219', '221', '223',
            '224', '227', '235', '237', '240', '241', '244', '248']
TEST_SET.sort()

ACCEL_LIST = [4] # 4, 6, 8]
LAMBDA_TV = 0 #_LIST = [1e-7, 1e-8, 1e-9, 1e-10]
NUM_ITER = 10000
LOSS_IN_KSP = False

def run_expmt(args):

    for file_id in args.file_id_list: 

        ksp = load_h5_qdess(file_id)

        # load, concat both echo slices
        idx_kx = ksp.shape[0] // 2 # want central slice in kx (axial) b/c we undersample in (ky,kz)
        ksp_echo1 = ksp[:,:,:,0,:].permute(3,0,1,2)[:, idx_kx, :, :]
        ksp_echo2 = ksp[:,:,:,1,:].permute(3,0,1,2)[:, idx_kx, :, :]
        ksp_orig = torch.cat((ksp_echo1, ksp_echo2), 0)

        for ACCEL in ACCEL_LIST:

            args.save_path='/bmrNAS/people/dvv/out_qdess/accel_{}x/echo_joint/loss_tv_tune/'\
                             .format(args.accel)
            sp = args.save_path
            if not os.path.exists(sp):
                os.makedirs(sp)
            if os.path.exists('{}MTR_{}_e1_dc_tv{}.npy'.format(sp, file_id, LAMBDA_TV)):
                continue

            # original masks created w central region 32x32 forced to 1's
            mask = torch.from_numpy(np.load('/home/vanveen/ConvDecoder/ipynb/masks/mask_poisson_disc_{}x.npy'.format(ACCEL)))
            
            # initialize network
            net, net_input, ksp_orig_ = init_convdecoder(ksp_orig, mask)

            # apply mask after rescaling k-space. want complex tensors dim (nc, ky, kz)
            ksp_masked = ksp_orig_ * mask
            img_masked = ifft_2d(ksp_masked)

            # fit network, get net output

            net, mse_wrt_ksp, mse_wrt_img = fit(
                ksp_masked=ksp_masked, img_masked=img_masked,
                net=net, net_input=net_input, mask2d=mask, num_iter=NUM_ITER,
                LOSS_IN_KSP=LOSS_IN_KSP, LAMBDA_TV=LAMBDA_TV)
            img_out = net(net_input.type(dtype)) # real tensor dim (2*nc, kx, ky)
            img_out = reshape_adj_channels_to_complex_vals(img_out[0]) # complex tensor dim (nc, kx, ky)
            
            # perform dc step
            ksp_est = fft_2d(img_out)
            ksp_dc = torch.where(mask, ksp_masked, ksp_est)

            # create data-consistent, ground-truth images from k-space
            img_1_dc = root_sum_squares(ifft_2d(ksp_dc[:8])).detach()
            img_1_gt = root_sum_squares(ifft_2d(ksp_orig[:8]))
            img_2_dc = root_sum_squares(ifft_2d(ksp_dc[8:])).detach()
            img_2_gt = root_sum_squares(ifft_2d(ksp_orig[8:]))
            
            # save results
            np.save('{}MTR_{}_e1_dc_tv{}.npy'.format(sp, file_id, LAMBDA_TV), img_1_dc)
            np.save('{}MTR_{}_e2_dc_tv{}.npy'.format(sp, file_id, LAMBDA_TV), img_2_dc)

            print('recon {}'.format(file_id)) 

return

def init_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0)
    #parser.add_argument('--lambda_tv', type=float, default=LAMBDA_TV)
    parser.add_argument('--accel', nargs='+', type=int, default=ACCEL_LIST)
    parser.add_argument('--file_id_list', nargs='+', default=TEST_SET)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = init_parser()

    torch.cuda.set_device(args.gpu_id)

    run_expmt(args)
