import torch 
import os, sys
import numpy as np
from torch.autograd import Variable

#from .helpers import np_to_var

dtype = torch.cuda.FloatTensor


def get_masked_measurements(slice_ksp, mask):
    ''' parameters: 
                slice_ksp: original, unmasked k-space measurements
                mask: mask used to downsample original k-space
        return:
                ksp_masked: masked measurements to fit
                img_masked: masked image, i.e. ifft(ksp_masked) '''
    
    raise NotImplementedError('need to fix this for processing w new fft/ifft')

    # mask the kspace
    #ksp_masked = apply_mask(np_to_tt(slice_ksp), mask=mask)
    ksp_masked = np_to_var(ksp_masked.data.cpu().numpy()).type(dtype)

    # perform ifft of masked kspace
    img_masked = ifft_2d(ksp_masked[0]).cpu().numpy()
    img_masked = reshape_complex_channels_to_be_adj(img_masked)
    img_masked = np_to_var(img_masked).type(dtype)
    
    return ksp_masked, img_masked

def data_consistency_iter(ksp, ksp_orig, mask1d, alpha=0.5):
    ''' apply dc step (ksp only) within gradient step
        i.e. replace vals of ksp w ksp_orig per mask1d 
        ksp, ksp_orig are torch tensors shape [1,15,x,y,2] '''

    # interpolate b/w ksp and ksp_orig according to mask1d
    ksp_dc = Variable(ksp.clone()).type(dtype) 
    ksp_dc[:,:,:,mask1d==1,:] = ksp_orig[:,:,:,mask1d==1,:] 
    
    return alpha*ksp_dc + (1-alpha)*ksp # as alpha increase, more weight on dc

def data_consistency(img_out, slice_ksp, mask1d):
    ''' perform data-consistency step given image 
        parameters:
                img_out: network output image
                slice_ksp: original k-space measurements 
        return:
                img_dc: data-consistent output image
                img_est: output image without data consistency '''
    
    raise NotImplementedError('redo this function based on new processing')

    #img_out = reshape_complex_channels_to_sep_dimn(img_out)

    # now get F*G(\hat{C}), i.e. estimated recon in k-space
    ksp_est = fft_2d(img_out) # now [30,x,y], prev  ([15,x,y,2]) 
    ksp_orig = np_to_tt(split_complex_vals(slice_ksp)) # [15,x,y,2]; slice_ksp (15,x,y) complex

    # replace estimated coeffs in k-space by original coeffs if it has been sampled
    mask1d = torch.from_numpy(np.array(mask1d, dtype=np.uint8)) # shape: torch.Size([368]) w 41 non-zero elements
    ksp_dc = ksp_est.clone().detach().cpu()
    ksp_dc[:,:,mask1d==1,:] = ksp_orig[:,:,mask1d==1,:]

    img_dc = recon_ksp_to_img(ksp_dc)
    img_est = recon_ksp_to_img(ksp_est.detach().cpu())
    
    return img_dc, img_est
