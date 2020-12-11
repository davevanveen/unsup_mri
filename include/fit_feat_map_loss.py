from torch.autograd import Variable
import torch
import copy
import numpy as np
import os, sys
import PIL
from PIL import Image
import torchvision
import time

from .transforms import *
sys.path.append('/home/vanveen/ConvDecoder/')
from utils.transform import fft_2d, ifft_2d, root_sum_squares, \
                            reshape_complex_vals_to_adj_channels, \
                            reshape_adj_channels_to_complex_vals
from utils.evaluate import normalize_img

dtype = torch.cuda.FloatTensor
mse = torch.nn.MSELoss()

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

def w_mse(x, y, c_wmse):
    ''' weighted mse by values of k-space '''
    ones = torch.ones(1).expand_as(x).type(dtype)
    mask = torch.where(x>1, c_wmse*x, ones)
    return torch.mean(mask * (x-y)**2)

class MSLELoss(torch.nn.Module):
    def __init__(self):
        super(MSLELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.log(criterion(x, y))
        return loss

def fit(ksp_masked, img_masked, net, net_input, mask2d, args, \
        hidden_size, lr=0.01):
    ''' fit a network to masked k-space measurement
        args:
            ksp_masked: masked k-space of a single slice. torch variable [1,C,H,W]
            img_masked: ifft(ksp_masked). torch variable [1,C,H,W]
            net: original network with randomly initiated weights
            net_input: randomly generated + scaled network input
            mask2d: mask for undersampling the ksp
            ksp_orig: orig ksp meas for data consistency step. torch tensor [15,x,y,2]
            num_iter: number of iterations to optimize network
            lr: learning rate
        returns:
            net: the best network, whose output would be in image space
            mse_wrt_ksp: mse(ksp_masked, fft(out)*mask) over each iteration
            mse_wrt_img: mse(img_masked, out) over each iteration
    '''            

    # initialize variables
    net_input = net_input.type(dtype)
    best_net = copy.deepcopy(net)
    best_mse = 10000.0
    mse_wrt_ksp, mse_wrt_img = np.zeros(args.num_iter), np.zeros(args.num_iter)
    
    p = [x for x in net.parameters()]
    optimizer = torch.optim.Adam(p, lr=lr,weight_decay=0)
    mse = torch.nn.MSELoss()

    # convert complex [nc,x,y] --> real [2*nc,x,y] to match w net output
    ksp_masked = reshape_complex_vals_to_adj_channels(ksp_masked).cuda()
    img_masked = reshape_complex_vals_to_adj_channels(img_masked)[None,:].cuda()
    mask2d = mask2d.cuda()

    # if performing feat map regulariz'n
    ksp_m_down_list, feat_map_mask_list, w_pix_list = \
                        get_meta_for_feat_map_loss(hidden_size, ksp_masked,
                                                   downsamp_method=args.downsamp_method,
                                                   weight_method=args.weight_method)

    for i in range(args.num_iter):
        def closure(): # execute this for each iteration (gradient step)

            optimizer.zero_grad()

            out, feat_map_list = net(net_input) # out is in img space
            out_ksp_masked = forwardm(out, mask2d).cuda() # convert img to ksp, apply mask

            #if DC_STEP: # ... see code inlay at bottom of file
            loss_ksp = mse(out_ksp_masked, ksp_masked)
            
            loss_feat_maps = get_loss_feat_maps(feat_map_list, ksp_m_down_list, 
                                                feat_map_mask_list, w_pix_list)
            loss_total = loss_ksp + args.alpha_fm * loss_feat_maps

            loss_total.backward(retain_graph=False)
            #loss_ksp.backward(retain_graph=False)

            # store loss over each iteration
            mse_wrt_ksp[i] = loss_ksp.data.cpu().numpy()
            loss_img = mse(out, img_masked) # loss in img space
            mse_wrt_img[i] = loss_img.data.cpu().numpy()

            return loss_total # original expmts returned loss_ksp...

        loss = optimizer.step(closure)

        # at each iteration, check if loss improves by 1%. if so, a new best net
        loss_val = loss.data
        if best_mse > 1.005*loss_val:
            best_mse = loss_val
            best_net = copy.deepcopy(net)

    return best_net, mse_wrt_ksp, mse_wrt_img

def forwardm(img, mask):
    ''' convert img --> ksp (must be complex for fft), apply mask
        input, output should have dim [2*nc,x,y] '''

    img = reshape_adj_channels_to_complex_vals(img[0]) 
    ksp = fft_2d(img).cuda()
    ksp_masked_ = ksp * mask
    
    return reshape_complex_vals_to_adj_channels(ksp_masked_)

def get_loss_feat_maps(feat_map_list, ksp_m_down_list, \
                       feat_map_mask_list, w_pix_list):
    ''' given feat_maps: hidden layer outputs of size [30,x_,y_], i.e. hidden_size
            ksp_m_down_list: downsampled ksp_masked according to each [x_,y_]
        compute weighted mse(feat_map_list, ksp_m_down_list) '''

    loss_feat = 0
    for idx, feat_map in enumerate(feat_map_list):

        # mask feat_map s.t. we don't penalized the masked indices in ksp_masked
        feat_map_m = feat_map * feat_map_mask_list[idx]

        # normalize ksp_m_down so we compute mse across similar distributions
        ksp_m_down_norm = normalize_img(feat_map_m, ksp_m_down_list[idx]).cuda()
        # overwrite masked indices which were non-zero'ed by normaliz'n
        ksp_m_down_norm = ksp_m_down_norm * feat_map_mask_list[idx]
        # compute mse, weighted to account for different num_pix in different layers
        loss_feat += w_pix_list[idx] * mse(feat_map_m, ksp_m_down_norm)

    return loss_feat

def get_meta_for_feat_map_loss(hidden_size, ksp_masked, \
                               downsamp_method='bicubic',
                               weight_method='all'):
    ''' given:
            ksp_masked: masked version of k-space
            hidden_size: list of [x_,y_], each the size of that hidden layer
            downsamp_method: e.g. 'nearest', 'bilinear', or 'bicubic'
            weight_method: whether to apply fm_loss on 'early', 'late', or 'all' layers
        return:
            ksp_m_down_list: downsampled ksp_masked according to each [x_,y_]
            feat_map_mask_list: binary mask for applying to feat_map s.t. we are not
                                penalizing against the zero columns in ksp_m_down
            w_pix: weights s.t. pixels from different layers are weighted equally '''

    if downsamp_method == 'bicubic':
        interp = PIL.Image.BICUBIC
    elif downsamp_method == 'bilinear':
        interp = PIL.Image.BILINEAR
    elif downsamp_method == 'nearest':
        interp = PIL.Image.NEAREST

    ksp_m_down_list, feat_map_mask_list = [], []

    for layer_size in hidden_size:

        # downsample ksp_masked according to each layer size
        downsamp = torchvision.transforms.Resize(layer_size, interpolation=interp)
        ksp_m_down = downsamp(ksp_masked)
        ksp_m_down_list.append(ksp_m_down)

        # create 1d mask w 1 if that row contains all zeros, 0 otw
        chan = ksp_m_down[0] # get single channel [x,y] from [2*nc,x,y]
        fm_mask = torch.where(torch.sum(chan, dim=0)==0, 0, 1).type(torch.uint8)
        feat_map_mask_list.append(fm_mask)

    pix_per_layer = np.array([x[0]*x[1] for x in hidden_size])
    w_pix_list = np.array([pix_per_layer.sum() / x for x in pix_per_layer])

    # if we want to weight early or later layers
    assert(len(w_pix_list) == 7)
    if weight_method == 'early':
        w_pix_list[-3:] = 0
    elif weight_method == 'late':
        w_pix_list[:4] = 0

    return ksp_m_down_list, feat_map_mask_list, w_pix_list
