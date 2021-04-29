import sys
import numpy as np
import torch

sys.path.append('/home/vanveen/ConvDecoder/')
from utils.data_io import get_mask, load_h5
from include.decoder_conv import init_convdecoder
from include.fit import fit
from utils.transform import fft_2d, ifft_2d, root_sum_squares, \
                            reshape_adj_channels_to_complex_vals, \
                            crop_center

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    torch.cuda.set_device(3)
else:
    dtype = torch.FloatTensor

dim = 320

file_id_list = ['1000273', '1000325', '1000464', '1000537', '1000818']#, \
                 #'1001140', '1001219']#, '1001338', '1001598', '1001533', '1001798']
file_id_list.sort()

SCALE_FAC = 0.1 # per 20201112_eval_fastmri_old_v_new.ipynb
NUM_ITER = 10000

path_out = '/bmrNAS/people/dvv/out_fastmri/ismrm/model5/'#.format(SCALE_FAC)


def run_expmt():
    for file_id in file_id_list:

        f, ksp_orig = load_h5(file_id)
        ksp_orig = torch.from_numpy(ksp_orig)

        mask = get_mask(ksp_orig)

        net, net_input, ksp_orig_, _ = init_convdecoder(ksp_orig, mask)

        ksp_masked = SCALE_FAC * ksp_orig_ * mask 
        img_masked = ifft_2d(ksp_masked)

        net, mse_wrt_ksp, mse_wrt_img = fit(
            ksp_masked=ksp_masked, img_masked=img_masked,
            net=net, net_input=net_input, mask2d=mask, num_iter=NUM_ITER)

        img_out, _ = net(net_input.type(dtype))
        img_out = reshape_adj_channels_to_complex_vals(img_out[0])
        ksp_est = fft_2d(img_out)
        ksp_dc = torch.where(mask, ksp_masked, ksp_est)

        img_est = crop_center(root_sum_squares(ifft_2d(ksp_est)).detach(), dim, dim)
        img_dc = crop_center(root_sum_squares(ifft_2d(ksp_dc)).detach(), dim, dim)
        img_gt = crop_center(root_sum_squares(ifft_2d(ksp_orig)), dim, dim)

        np.save('{}{}_est.npy'.format(path_out, file_id), img_est)
        np.save('{}{}_dc.npy'.format(path_out, file_id), img_dc)
        np.save('{}{}_gt.npy'.format(path_out, file_id), img_gt)

if __name__ == '__main__':
    run_expmt()
