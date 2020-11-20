import os, sys
import torch
import numpy as np
import random, string

sys.path.append('/home/vanveen/ConvDecoder/')
from utils.data_io import load_h5, get_mask
from include.decoder_conv import init_convdecoder
from include.fit_feat_map_loss import fit
from utils.transform import fft_2d, ifft_2d, root_sum_squares, \
                            reshape_complex_vals_to_adj_channels, \
                            reshape_adj_channels_to_complex_vals, \
                            crop_center
from utils.evaluate import calc_metrics

#if torch.cuda.is_available():
#    torch.backends.cudnn.enabled = True
#    torch.backends.cudnn.benchmark = True
#    dtype = torch.cuda.FloatTensor
#    torch.cuda.set_device(0)
#else:
#    dtype = torch.FloatTensor


def train(args):

    print(args)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu_id)
    else:
        dtype = torch.FloatTensor

    DIM = 320
    SCALE_FAC = 0.1

    file_id_list = ['1000273', '1000325', '1000464']#, '1000007', '1000537', '1000818', \
    #                  '1001140', '1001219', '1001338', '1001598', '1001533', '1001798']
    file_id_list.sort()

    path_out = '/bmrNAS/people/dvv/out_fastmri/expmt_fm_loss/'
    trial_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    for file_id in file_id_list:

        f, ksp_orig = load_h5(file_id)
        ksp_orig = torch.from_numpy(ksp_orig)

        mask = get_mask(ksp_orig)
        
        net, net_input, ksp_orig_, hidden_size = init_convdecoder(ksp_orig, mask)

        ksp_masked = SCALE_FAC * ksp_orig_ * mask
        img_masked = ifft_2d(ksp_masked)

        net, mse_wrt_ksp, mse_wrt_img = fit(
            ksp_masked=ksp_masked, img_masked=img_masked,
            net=net, net_input=net_input, mask2d=mask, args=args,
            hidden_size=hidden_size)


        img_out = net(net_input.type(dtype))
        img_out = img_out[0] if type(img_out) is tuple else img_out

        img_out = reshape_adj_channels_to_complex_vals(img_out[0])
        ksp_est = fft_2d(img_out)
        ksp_dc = torch.where(mask, ksp_masked, ksp_est)

        img_est = crop_center(root_sum_squares(ifft_2d(ksp_est)).detach(), DIM, DIM)
        img_dc = crop_center(root_sum_squares(ifft_2d(ksp_dc)).detach(), DIM, DIM)
        img_gt = crop_center(root_sum_squares(ifft_2d(ksp_orig)), DIM, DIM)
        
        _, _, ssim_est, psnr_est = calc_metrics(np.array(img_est), np.array(img_gt))
        _, _, ssim_dc, psnr_dc = calc_metrics(np.array(img_dc), np.array(img_gt))

        ssim_est, psnr_est = np.round(ssim_est, 4), np.round(psnr_est, 4)
        ssim_dc, psnr_dc = np.round(ssim_dc, 4), np.round(psnr_dc, 4)
        
        # get output + params, write to new line in csv
        line = [trial_id, file_id, ssim_dc, psnr_dc, ssim_est, psnr_est, \
                args.alpha_fm, args.num_iter, args.iter_start_fm_loss, \
                args.weight_method, args.downsamp_method]
        line = [str(x) for x in line] # convert all entries to str
        f = open(args.csv_fn, 'a')
        f.write(','.join(line) + '\n')
        f.close()

        fn_out = '{}{}_{}'.format(path_out, trial_id, file_id)
        np.save('{}_est.npy'.format(fn_out), img_est)
        np.save('{}_dc.npy'.format(fn_out), img_dc)
