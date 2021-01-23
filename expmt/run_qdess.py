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
    torch.cuda.set_device(3)

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
NUM_SAMPS = 4 # len(test_set)

NUM_ITER = 10000
ACCEL_LIST = [4] # 4, 6, 8]

def run_expmt():

    for fn in test_set[:NUM_SAMPS]: 

       # load data
        f = h5py.File(path_in + fn, 'r')
        try:
            ksp = torch.from_numpy(f['kspace'][()])
        except KeyError:
            print('No kspace in file {} w keys {}'.format(fn, f.keys()))
            f.close()
            continue
        f.close()

        # NOTE: if change to echo2, must manually change path nomenclature
        ksp_vol = ksp[:,:,:,1,:].permute(3,0,1,2) # get echo1, reshape to be (nc, kx, ky, kz)

        # get central slice in kx, i.e. axial plane b/c we undersample in (ky, kz)
        idx_kx = ksp_vol.shape[1] // 2
        ksp_orig = ksp_vol[:, idx_kx, :, :]

        for ACCEL in ACCEL_LIST:
           
            path_out = '/bmrNAS/people/dvv/out_qdess/accel_{}x/echo2/'.format(ACCEL)
            
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
            img_dc = root_sum_squares(ifft_2d(ksp_dc)).detach()
            img_gt = root_sum_squares(ifft_2d(ksp_orig))

            # save results
            samp = fn.split('.h5')[0] 
            np.save('{}{}_dc.npy'.format(path_out, samp), img_dc)
            np.save('{}{}_gt.npy'.format(path_out, samp), img_gt)

            print('recon {}'.format(samp)) 

    return


if __name__ == '__main__':
    run_expmt()
