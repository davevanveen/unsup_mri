import os, sys
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append('/home/vanveen/ConvDecoder/')
from utils.data_io import load_h5
from utils.helpers import num_params
from include.decoder_conv import init_convdecoder
from include.fit import fit
from include.subsample import MaskFunc
from utils.evaluate import calc_metrics
from utils.transform import fft_2d, ifft_2d, root_sum_squares, \
                            reshape_complex_vals_to_adj_channels, \
                            reshape_adj_channels_to_complex_vals, \
                            crop_center

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    torch.cuda.set_device(0)
else:
    dtype = torch.FloatTensor

def get_mask(ksp_orig, center_fractions=[0.07], accelerations=[4]):
    ''' simplified version of get_masks() in utils.helpers -- return only a 1d mask in torch tensor '''
        
    mask_func = MaskFunc(center_fractions=center_fractions, \
                             accelerations=accelerations)
    
    # note: had to swap dims to be compatible w facebook's MaskFunc class
    mask_shape = (1, ksp_orig.shape[2], ksp_orig.shape[1])
    
    mask = mask_func(mask_shape, seed=0)

    return mask[0,:,0].type(torch.uint8)

dim = 320

file_id_list = ['1000273', '1000325', '1000464', '1000007', '1000537', '1000818', \
                 '1001140', '1001219']#, '1001338', '1001598', '1001533', '1001798']
file_id_list.sort()

#scale_fac_list = [1, 0.1, 0.05, 0.01]
scale_fac = [0.1] # set this based on results in 20201112_eval_fastmri_old_v_new.ipynb

for scale_fac in scale_fac_list:

    path_out = '/bmrNAS/people/dvv/out_fastmri/new_pytorch1.7/sf{}/'.format(scale_fac)

    for file_id in file_id_list:

        if os.path.exists('{}{}_dc.npy'.format(path_out, file_id)):
            continue

        f, ksp_orig = load_h5(file_id)
        ksp_orig = torch.from_numpy(ksp_orig)

        mask = get_mask(ksp_orig)

        net, net_input, ksp_orig_ = init_convdecoder(ksp_orig, mask)

        ksp_masked = scale_fac * ksp_orig_ * mask # previously had multiplier of 0.5
        img_masked = ifft_2d(ksp_masked)

        net, mse_wrt_ksp, mse_wrt_img = fit(
            ksp_masked=ksp_masked, img_masked=img_masked,
            net=net, net_input=net_input, mask2d=mask, num_iter=10000)

        img_out = net(net_input.type(dtype))[0]
        img_out = reshape_adj_channels_to_complex_vals(img_out)
        ksp_est = fft_2d(img_out)
        ksp_dc = torch.where(mask, ksp_masked, ksp_est)

        img_est = crop_center(root_sum_squares(ifft_2d(ksp_est)).detach(), dim, dim)
        img_dc = crop_center(root_sum_squares(ifft_2d(ksp_dc)).detach(), dim, dim)
        img_gt = crop_center(root_sum_squares(ifft_2d(ksp_orig)), dim, dim)
        print('note: use unscaled ksp_orig to make gt -- different from old fastmri processing')

        np.save('{}{}_est.npy'.format(path_out, file_id), img_est)
        np.save('{}{}_dc.npy'.format(path_out, file_id), img_dc)
        np.save('{}{}_gt.npy'.format(path_out, file_id), img_gt)
