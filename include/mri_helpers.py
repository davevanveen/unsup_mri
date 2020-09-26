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

from utils.transform import np_to_tt, np_to_var, apply_mask, ifft_2d, fft_2d, \
                            reshape_complex_channels_to_sep_dimn, \
                            reshape_complex_channels_to_be_adj, \
                            split_complex_vals, recon_ksp_to_img

dtype = torch.cuda.FloatTensor


def get_masked_measurements(slice_ksp, mask):
    ''' parameters: 
                slice_ksp: original, unmasked k-space measurements
                mask: mask used to downsample original k-space
        return:
                ksp_masked: masked measurements to fit
                img_masked: masked image, i.e. ifft(ksp_masked) '''

    # mask the kspace
    ksp_masked = apply_mask(np_to_tt(slice_ksp), mask=mask)
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
    #ksp_dc[:,:,:,mask1d==1,:] = ksp_orig[:,:,:,mask1d==1,:] 
    ksp_dc[:,:,:,:,:] = ksp_orig[:,:,:,:,:] 
    
    return alpha*ksp_dc + (1-alpha)*ksp # as alpha increase, more weight on dc

def data_consistency(img_out, slice_ksp, mask1d):
    ''' perform data-consistency step given image 
        parameters:
                img_out: network output image
                slice_ksp: original k-space measurements 
        return:
                img_dc: data-consistent output image
                img_est: output image without data consistency '''
    
    img_out = reshape_complex_channels_to_sep_dimn(img_out)

    # now get F*G(\hat{C}), i.e. estimated recon in k-space
    ksp_est = fft_2d(img_out) # ([15,x,y,2])
    ksp_orig = np_to_tt(split_complex_vals(slice_ksp)) # [15,x,y,2]; slice_ksp (15,x,y) complex

    # replace estimated coeffs in k-space by original coeffs if it has been sampled
    mask1d = torch.from_numpy(np.array(mask1d, dtype=np.uint8)) # shape: torch.Size([368]) w 41 non-zero elements
    ksp_dc = ksp_est.clone().detach().cpu()
    ksp_dc[:,:,mask1d==1,:] = ksp_orig[:,:,mask1d==1,:]

    img_dc = recon_ksp_to_img(ksp_dc)
    img_est = recon_ksp_to_img(ksp_est.detach().cpu())
    
    return img_dc, img_est

def forwardm(img, mask):
    # img has dimension (2*num_slices, x,y)
    # output has dimension (1, num_slices, x, y, 2)
    mask = np_to_var(mask)[0].type(dtype)
    s = img.shape
    ns = int(s[1]/2) # number of slices
    fimg = Variable( torch.zeros( (s[0],ns,s[2],s[3],2 ) ) ).type(dtype)
    for i in range(ns):
        fimg[0,i,:,:,0] = img[0,2*i,:,:]
        fimg[0,i,:,:,1] = img[0,2*i+1,:,:]
    Fimg = transform.fft2(fimg) # dim: (1,num_slices,x,y,2)
    for i in range(ns):
        Fimg[0,i,:,:,0] *= mask
        Fimg[0,i,:,:,1] *= mask
    return Fimg

def get_scale_factor(net, num_channels, in_size, slice_ksp, scale_out=1, scale_type='norm'):
    ''' return net_input, e.g. tensor w values sampled uniformly on [0,1]

        return scaling factor, i.e. difference in magnitudes scaling b/w:
        original image and random image of network output = net(net_input) '''
    #TODO: make this two separate functions - get_scale_factor and get_net_input    

    # create net_input, e.g. tensor with values sampled uniformly on [0,1]
    shape = [1,num_channels, in_size[0], in_size[1]]
    net_input = Variable(torch.zeros(shape)).type(dtype)
    net_input.data.uniform_()

    # generate random image
    try:
        out_chs = net(net_input.type(dtype), scale_out=scale_out).data.cpu().numpy()[0]
    except:
        out_chs = net(net_input.type(dtype)).data.cpu().numpy()[0]
    out_imgs = channels2imgs(out_chs)
    out_img_tt = transform.root_sum_of_squares(torch.tensor(out_imgs), dim=0)

    ### get norm of least-squares reconstruction
    ksp_tt = transform.to_tensor(slice_ksp)
    orig_tt = transform.ifft2(ksp_tt)   # apply ifft get the complex image
    orig_imgs_tt = transform.complex_abs(orig_tt)   # compute absolute value to get a real image
    orig_img_tt = transform.root_sum_of_squares(orig_imgs_tt, dim=0)
    orig_img_np = orig_img_tt.cpu().numpy()
    
    if scale_type == "norm":
        scale = np.linalg.norm(out_img_tt) / np.linalg.norm(orig_img_np)
    if scale_type == "mean":
        scale = (out_img_tt.mean() / orig_img_np.mean()).numpy()[np.newaxis][0]
    
    return scale, net_input

def lsreconstruction(measurement,mode='both'):
    ''' given measurement of dimn (1, num_slices, x, y, 2), 
        take ifft and return either the
        real components, imag components, or combined magnitude '''
    
    fimg = transform.ifft2(measurement)
    #print("real/img parts: ", torch.norm(fimg[:,:,:,:,0]), torch.norm(fimg[:,:,:,:,1]))
    
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
