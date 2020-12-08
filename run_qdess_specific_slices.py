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
    torch.cuda.set_device(1)

test_set = [
 'MTR_065.h5',
 'MTR_066.h5']

# axial slices of interest
idx_kx_dict = {'MTR_065.h5': [219, 220, 221, 372, 373, 374],
               'MTR_066.h5': [277, 278, 279, 280]}

def run_expmt():

    path_in = '/bmrNAS/people/arjun/data/qdess_knee_2020/files_recon_calib-16/'
    #files = [f for f in listdir(path_in) if isfile(join(path_in, f))]
    #files.sort()
    #NUM_SAMPS = 10 # number of samples to recon
       
    NUM_ITER = 10000
    ACCEL_LIST = [4]#, 6, 8]

    for fn in test_set: #files[:NUM_SAMPS]:

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
        ksp_vol = ksp[:,:,:,0,:].permute(3,0,1,2) # get echo1, reshape to be (nc, kx, ky, kz)

        # get central slice in kx, i.e. axial plane b/c we undersample in (ky, kz)
        idx_kx_list = idx_kx_dict[fn]
        
        for idx_kx in idx_kx_list:

            ksp_orig = ksp_vol[:, idx_kx, :, :]

            for ACCEL in ACCEL_LIST:
               
                path_out = '/bmrNAS/people/dvv/out_qdess/specific_slices/accel_{}x/echo1/'.format(ACCEL)
                
                # original masks created w central region 32x32 forced to 1's
                mask = torch.from_numpy(np.load('ipynb/masks/mask_poisson_disc_{}x.npy'.format(ACCEL)))

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
                samp = fn.split('.h5')[0] #+ '_echo2' 
                np.save('{}{}_dc_slice{}.npy'.format(path_out, samp, idx_kx), img_dc)
                np.save('{}{}_gt_slice{}.npy'.format(path_out, samp, idx_kx), img_gt)

                print('recon {}'.format(samp)) 

    return


if __name__ == '__main__':
    run_expmt()
