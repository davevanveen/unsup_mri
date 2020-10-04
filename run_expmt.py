import os, sys
import numpy as np
import torch

from utils.data_io import load_h5, load_output, save_output, \
                            expmt_already_generated
from utils.transform import np_to_tt, split_complex_vals, recon_ksp_to_img
from utils.helpers import num_params, get_masks
from include.decoder_conv import init_convdecoder
from include.mri_helpers import get_scale_factor, get_masked_measurements, \
                                data_consistency
from include.fit import fit
from utils.evaluate import calc_metrics

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    torch.cuda.set_device(1)
else:
    dtype = torch.FloatTensor


def run_expmt():
    
    img_dc_list, img_est_list, img_gt_list = [], [], []
    mse_wrt_ksp_list, mse_wrt_img_list = [], []
    ssim_list, psnr_list = [], []

    file_id_list = ['1000273', '1000325', '1000464', '1000007', '1000537', '1000818']#, \
    #                 '1001140', '1001219', '1001338', '1001598', '1001533', '1001798']
    NUM_ITER_LIST = [100, 1000, 10000]
    ALPHA_LIST = [0, 0.25, 0.5]#, 0.25, 0.5, 0.75]
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
                img_gt = recon_ksp_to_img(slice_ksp) # must do this after slice_ksp is scaled
                img_gt_list.append(img_gt) # could do this once per loop

                if expmt_already_generated(file_id, NUM_ITER, DC_STEP, ALPHA):
                    img_dc_list.append(load_output(file_id, NUM_ITER, DC_STEP, ALPHA))
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
