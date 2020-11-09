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
from utils.helpers import num_params#, get_masks
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

def run_expmt():

    path = '/bmrNAS/people/arjun/data/qdess_knee_2020/files_recon_calib-16/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files.sort()
    NUM_SAMPS = 25 # number of samples to recon
       
    NUM_ITER = 10000

    # load mask, force central CxC pixels in mask to be 1
    mask = torch.from_numpy(np.load('mask_3d.npy')
    mask = abs(mask).type(torch.uint8)
    idx_y, idx_z = mask.shape[0] // 2, mask.shape[1] // 2
    C = 32
    mask[idx_y-C:idx_y+C, idx_z-C:idx_z+C] = 1

    for fn in files[:NUM_SAMPS]:

        # load data
        f = h5py.File(path + fn, 'r')
        ksp = torch.from_numpy(f['kspace'][()])
        ksp_vol = ksp[:,:,:,0,:].permute(3,0,1,2) # get echo1, reshape to be (nc, kx, ky, kz)

        # get central slice in kx, i.e. axial plane b/c we undersample in (ky, kz)
        idx_kx = ksp_vol.shape[1] // 2
        ksp_orig = ksp_vol[:, idx_kx, :, :]

        # initialize network
        net, net_input, ksp_orig_ = init_convdecoder(ksp_orig, mask)

        # apply mask after rescaling k-space. want complex tensors dim (nc, ky, kz)
        ksp_masked = ksp_orig_ * mask
        img_masked = ifft_2d(ksp_masked)

        # fit network, get net output
        net, mse_wrt_ksp, mse_wrt_img = fit(
            ksp_masked=ksp_masked, img_masked=img_masked,
            net=net, net_input=net_input, mask2d=mask, num_iter=NUM_ITER)
        img_out = net(net_input.type(dtype))[0] # real tensor dim (2*nc, kx, ky)
        img_out = reshape_adj_channels_to_complex_vals(img_out) # complex tensor dim (nc, kx, ky)
        
        # perform dc step
        ksp_est = fft_2d(img_out)
        ksp_dc = torch.where(mask, ksp_masked, ksp_est) # dc step

        # create images from k-space
        







def run_expmt():
    
    #file_id_list = ['1000273', '1000325', '1000464', '1000007', '1000537', '1000818', \
    #                '1001140', '1001219', '1001338', '1001598', '1001533', '1001798']
    file_id_list = ['1001533', '1001798'] # two samples w biggest delta b/w ALPHA=0, ALPHA=.25
    NUM_ITER_LIST = [10000]
    ALPHA_LIST = [0.125, 0.375]#, 0.25, 0.5, 0.75]
    DC_STEP = True

    for idx, file_id in enumerate(file_id_list):

        f, slice_ksp = load_h5(file_id) # load full mri measurements
        print('file_id: {}'.format(file_id))

        mask, mask2d, mask1d = get_masks(f, slice_ksp) # load mask + variants, M
        mask1d_ = torch.from_numpy(np.array(mask1d, dtype=np.uint8)) # for dc step

        for NUM_ITER in NUM_ITER_LIST:
            for ALPHA in ALPHA_LIST:

                # initialize net, net input seed, and scale slice_ksp accordingly
                net, net_input, slice_ksp = init_convdecoder(slice_ksp, mask)
                #img_gt = recon_ksp_to_img(slice_ksp) # must do this after slice_ksp is scaled

                if expmt_already_generated(file_id, NUM_ITER, DC_STEP, ALPHA):
                    continue

                # for dc step - must do this after scaling slice_ksp
                ksp_orig = np_to_tt(split_complex_vals(slice_ksp))[None, :].type(dtype) #[1,15,640,368,2]

                # apply mask to measurements for fitting model
                ksp_masked, img_masked = get_masked_measurements(slice_ksp, mask)

                net, mse_wrt_ksp, mse_wrt_img = fit(
                    ksp_masked=ksp_masked, img_masked=img_masked,
                    net=net, net_input=net_input, mask2d=mask2d,
                    mask1d=mask1d_, ksp_orig=ksp_orig, DC_STEP=DC_STEP, alpha=ALPHA,
                    img_ls=None, num_iter=NUM_ITER, dtype=dtype)

                img_out = net(net_input.type(dtype))[0] # estimate image \hat{x} = G(\hat{C})

                img_dc, _ = data_consistency(img_out, slice_ksp, mask1d)
                save_output(img_dc, mse_wrt_ksp, mse_wrt_img, \
                            file_id, NUM_ITER, DC_STEP, ALPHA)

if __name__ == '__main__':
    run_expmt()
