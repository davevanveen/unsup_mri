import os, sys
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import h5py
import sigpy
from sigpy.mri.samp import poisson
import torch

sys.path.append('/home/vanveen/ConvDecoder/')
from utils.data_io import num_params
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
    torch.cuda.set_device(2)

path_in = '/bmrNAS/people/arjun/data/qdess_knee_2020/files_recon_calib-16/'
test_set = [
             'MTR_030.h5', 'MTR_034.h5', 'MTR_048.h5', 'MTR_052.h5',
             'MTR_065.h5', 'MTR_066.h5', 'MTR_005.h5', 'MTR_006.h5',
             'MTR_080.h5', 'MTR_096.h5',
             'MTR_099.h5', 'MTR_120.h5', 'MTR_144.h5', 'MTR_156.h5',
             'MTR_158.h5', 'MTR_173.h5', 'MTR_176.h5', 'MTR_178.h5',
             'MTR_188.h5', 'MTR_196.h5', 'MTR_198.h5', 'MTR_199.h5',
             'MTR_218.h5', 'MTR_219.h5', 'MTR_221.h5', 'MTR_223.h5',
             'MTR_224.h5', 'MTR_227.h5', 'MTR_235.h5', 'MTR_237.h5',
             'MTR_240.h5', 'MTR_241.h5', 'MTR_244.h5', 'MTR_248.h5'
]
test_set.sort()

NUM_ITER = 10000
ACCEL_LIST = [8] # 4, 6, 8]

def run_expmt():

    for fn in test_set: 

       # load data
        f = h5py.File(path_in + fn, 'r')
        try:
            ksp = torch.from_numpy(f['kspace'][()])
        except KeyError:
            print('No kspace in file {} w keys {}'.format(fn, f.keys()))
            f.close()
            continue
        f.close()

        # load, concat both echo slices
        idx_kx = ksp.shape[0] // 2 # want central slice in kx (axial) b/c we undersample in (ky,kz)
        ksp_echo1 = ksp[:,:,:,0,:].permute(3,0,1,2)[:, idx_kx, :, :]
        ksp_echo2 = ksp[:,:,:,1,:].permute(3,0,1,2)[:, idx_kx, :, :]
        ksp_orig = torch.cat((ksp_echo1, ksp_echo2), 0)

        for ACCEL in ACCEL_LIST:
           
            path_out = '/bmrNAS/people/dvv/out_qdess/accel_{}x/echo_joint/'.format(ACCEL)
            
            # original masks created w central region 32x32 forced to 1's
            mask = torch.from_numpy(np.load('/home/vanveen/ConvDecoder/ipynb/masks/mask_poisson_disc_{}x.npy'.format(ACCEL)))

            # initialize network
            net, net_input, ksp_orig_, _ = init_convdecoder(ksp_orig, mask)

            # apply mask after rescaling k-space. want complex tensors dim (nc, ky, kz)
            ksp_masked = ksp_orig_ * mask
            img_masked = ifft_2d(ksp_masked)

            # fit network, get net output
            net, mse_wrt_ksp, mse_wrt_img = fit(
                ksp_masked=ksp_masked, img_masked=img_masked,
                net=net, net_input=net_input, mask2d=mask, num_iter=NUM_ITER)
            img_out, _ = net(net_input.type(dtype)) # real tensor dim (2*nc, kx, ky)
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
            samp = fn.split('.h5')[0] #+ '_echo2' 
            np.save('{}{}_e1-joint-recon_dc.npy'.format(path_out, samp), img_1_dc)
            np.save('{}{}_e1_gt.npy'.format(path_out, samp), img_1_gt)
            np.save('{}{}_e2-joint-recon_dc.npy'.format(path_out, samp), img_2_dc)
            np.save('{}{}_e2_gt.npy'.format(path_out, samp), img_2_gt)

            print('recon {}'.format(samp)) 

    return


if __name__ == '__main__':
    run_expmt()
