from torch.autograd import Variable
import torch
import copy
import numpy as np
import os, sys
import time

from .transforms import *
sys.path.append('/home/vanveen/ConvDecoder/')
from utils.transform import fft_2d, ifft_2d, root_sum_squares, \
                            reshape_complex_vals_to_adj_channels, \
                            reshape_adj_channels_to_complex_vals
dtype = torch.cuda.FloatTensor

def sqnorm(a):
    return np.sum( a*a )

def get_distances(initial_maps,final_maps):
    results = []
    for a,b in zip(initial_maps,final_maps):
        res = sqnorm(a-b)/(sqnorm(a) + sqnorm(b))
        results += [res]
    return(results)

def get_weights(net):
    weights = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            weights += [m.weight.data.cpu().numpy()]
    return weights

class MSLELoss(torch.nn.Module):
    def __init__(self):
        super(MSLELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.log(criterion(x, y))
        return loss

def fit(ksp_masked, img_masked, net, net_input, mask2d, 
        mask1d=None, ksp_orig=None, DC_STEP=False, alpha=0.5,
        num_iter=5000, lr=0.01, img_ls=None, dtype=torch.cuda.FloatTensor, 
        LAMBDA_TV=1e-8):
    ''' fit a network to masked k-space measurement
        args:
            ksp_masked: masked k-space of a single slice. torch variable [1,C,H,W]
            img_masked: ifft(ksp_masked). torch variable [1,C,H,W]
            net: original network with randomly initiated weights
            net_input: randomly generated + scaled network input
            mask2d: 2D mask for undersampling the ksp
            mask1d: 1D mask for data consistency step. boolean torch tensor size [y]
            ksp_orig: orig ksp meas for data consistency step. torch tensor [15,x,y,2]
            DC_STEP: (boolean) do data consistency step during network fit
            alpha: control weight to which we enforce data consistency,
                   e.g. ksp_out = alpha*ksp_dc + (1-alpha)*ksp_in, 0 <= alpha < 1
            num_iter: number of iterations to optimize network
            lr: learning rate
            img_ls: least-squares image of unmasked k-space, i.e. ifft(ksp_full)
                    only needed to compute ssim, psnr over each iteration
        returns:
            net: the best network, whose output would be in image space
            mse_wrt_ksp: mse(ksp_masked, fft(out)*mask) over each iteration
            mse_wrt_img: mse(img_masked, out) over each iteration
    '''            

    # initialize variables
    if img_ls is not None or net_input is None: 
        raise NotImplementedError('incorporate original code here')
    if alpha < 0 or alpha >= 1:
        raise ValueError('alpha must be non-negative and strictly less than 1')
    net_input = net_input.type(dtype)
    best_net = copy.deepcopy(net)
    best_mse = 10000.0
    mse_wrt_ksp, mse_wrt_img = np.zeros(num_iter), np.zeros(num_iter)
    
    p = [x for x in net.parameters()]
    optimizer = torch.optim.Adam(p, lr=lr,weight_decay=0)
    mse = torch.nn.MSELoss()

    # convert complex [nc,x,y] --> real [2*nc,x,y] to match w net output
    ksp_masked = reshape_complex_vals_to_adj_channels(ksp_masked).cuda()
    img_masked = reshape_complex_vals_to_adj_channels(img_masked)[None,:].cuda()
    mask2d = mask2d.cuda()

    for i in range(num_iter):
        def closure(): # execute this for each iteration (gradient step)

            optimizer.zero_grad()

            out = net(net_input) # out is in img space

            #if LOSS_IN_KSP: # perform loss in k-space
            #    out_ksp_masked = forwardm(out, mask2d).cuda() # convert img to ksp, apply mask
            #    loss_ksp = mse(out_ksp_masked, ksp_masked)
            #    loss_total = loss_ksp
            #else:
            out_img_masked = forwardm_img(out, mask2d) # img-->ksp, mask, convert to img
            loss_img = mse(out_img_masked, img_masked)
            loss_tv = (torch.sum(torch.abs(out_img_masked[:,:,:,:-1] - \
                                           out_img_masked[:,:,:,1:])) \
                     + torch.sum(torch.abs(out_img_masked[:,:,:-1,:] - \
                                           out_img_masked[:,:,1:,:])))
            loss_total = loss_img + LAMBDA_TV * loss_tv
            
            loss_total.backward(retain_graph=False)

            # mse_wrt_ksp[i] = loss_ksp.data.cpu().numpy() # store loss over each iteration
            
            return loss_total

        loss = optimizer.step(closure)

        # at each iteration, check if loss improves by 1%. if so, a new best net
        loss_val = loss.data
        if best_mse > 1.005*loss_val:
            best_mse = loss_val
            best_net = copy.deepcopy(net)
   
    return best_net, mse_wrt_ksp, mse_wrt_img

def forwardm_img(img, mask):
    ''' convert img --> ksp (must be complex for fft), apply mask
        convert back to img. input dim [2*nc,x,y], output dim [1,2*nc,x,y] '''

    img = reshape_adj_channels_to_complex_vals(img[0])
    ksp = fft_2d(img).cuda()
    ksp_masked_ = ksp * mask
    img_masked_ = ifft_2d(ksp_masked_)

    return reshape_complex_vals_to_adj_channels(img_masked_)[None, :]

def forwardm(img, mask):
    ''' convert img --> ksp (must be complex for fft), apply mask
        input, output should have dim [2*nc,x,y] '''

    img = reshape_adj_channels_to_complex_vals(img[0]) 
    ksp = fft_2d(img).cuda()
    ksp_masked_ = ksp * mask
    
    return reshape_complex_vals_to_adj_channels(ksp_masked_)
