import torch 
import torchvision
import os, sys
import numpy as np
from PIL import Image
import PIL
import copy
from torch.autograd import Variable
import matplotlib.pyplot as plt

from . import transforms as transform
from .helpers import np_to_var

from utils.transform import np_to_tt, np_to_var, ifft_2d, fft_2d, \
                            reshape_complex_channels_to_sep_dimn, \
                            reshape_complex_channels_to_be_adj, \
                            split_complex_vals, recon_ksp_to_img, \
                            reshape_adj_channels_to_be_complex

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

def forwardm(img, mask):
    ''' convert img --> ksp, apply mask
        img + output each have dimension (2*num_slices, x,y) '''

    img = reshape_adj_channels_to_be_complex(img[0]) # [1,2*nc,x,y] real --> [nc,x,y] complex
    ksp = fft_2d(img).type(dtype) 
    ksp_masked = ksp * mask

    return ksp_masked

def lsreconstruction(measurement,mode='both'):
    ''' given measurement of dimn (1, num_slices, x, y, 2), 
        take ifft and return either the
        real components, imag components, or combined magnitude '''
    
    fimg = ifft_2d(measurement)
    raise TypeError('new ifft_2d() outputs (-2,-1) as spatial dimensions - \
                    might cause issue with below code')
    
    if mode == 'both':
        return torch.sqrt(fimg[:,:,:,:,0]**2 + fimg[:,:,:,:,1]**2)
    elif mode == 'real':
        return torch.tensor(fimg[:,:,:,:,0]) #torch.sqrt(fimg[:,:,:,:,0]**2)
    elif mode == 'imag':
        return torch.sqrt(fimg[:,:,:,:,1]**2)

def channels2imgs(out): #TODO: replace this function via utils.transform.py
    sh = out.shape
    chs = int(sh[0]/2)
    imgs = np.zeros( (chs,sh[1],sh[2]) )
    for i in range(chs):
        imgs[i] = np.sqrt( out[2*i]**2 + out[2*i+1]**2 )
    return imgs
