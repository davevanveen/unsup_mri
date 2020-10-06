from torch.autograd import Variable
import torch
import torch.optim
import copy
import numpy as np
import time
from scipy.linalg import hadamard
from skimage.metrics import structural_similarity as ssim

from .helpers import *
from .mri_helpers import * #forwardm, 
from .transforms import *

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

def fit(ksp_masked, img_masked, net, net_input, mask2d, 
        mask1d=None, ksp_orig=None, DC_STEP=False, alpha=0.5,
        num_iter=5000, lr=0.01, img_ls=None, dtype=torch.cuda.FloatTensor, 
        c_wmse=1):
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
    
        notes on the original code via heckel paper
            - initialized out_grads, out_weights, + gave opt_input option
            - gave option to generate net_input via upsampling, reshaping, scaling
            - within closure()
                - adjusted scaling of input
                - applied mask when computing loss = mse( , )
                - provided option for showing imgs, plotting stuff, and outputting weights
                - if opt_input=True, would save best_net_input
                - computed ssim, psnr, across iterations w unmasked least-squares recon
                    - requires img_ls argument
            - returned many additional variables
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

    for i in range(num_iter):
        def closure(): # execute this for each iteration (gradient step)
            
            optimizer.zero_grad()
            
            out = net(net_input) # out is in image space

            # forwardm(): converts img to ksp, apply mask, and return the masked ksp
                # TODO: compare forwardm to utils.transform.apply_mask()
                # add forwardm().half() to do with half precision
            out_ksp_masked = forwardm(out, mask2d)

            if DC_STEP:
                out_ksp_dc = data_consistency_iter(ksp=out_ksp_masked, 
                                                   ksp_orig=ksp_orig, mask1d=mask1d, 
                                                   alpha=alpha)
                loss_ksp = mse(out_ksp_dc, ksp_masked)
            else:
                loss_ksp = w_mse(out_ksp_masked, ksp_masked, c_wmse) # loss wrt masked k-space
            
            # TODO: why do we backprop on loss_ksp and not loss_img? think about this
            loss_ksp.backward(retain_graph=False)
            
            mse_wrt_ksp[i] = loss_ksp.data.cpu().numpy()

            # loss in image space
            loss_img = mse(out, img_masked.type(dtype))
            mse_wrt_img[i] = loss_img.data.cpu().numpy()

            return loss_ksp   
 
        loss = optimizer.step(closure)

        # at each iteration, check if loss improves by 1%. if so, a new best net
        loss_val = loss.data
        if best_mse > 1.005*loss_val:
            best_mse = loss_val
            best_net = copy.deepcopy(net)
    
    return best_net, mse_wrt_ksp, mse_wrt_img


        
def fit_multiple(net,
        imgs, # list of images [ [1, color channels, W, H] ] 
        num_channels,
        num_iter = 5000,
        lr = 0.01,
        find_best=False,
        upsample_mode="bilinear",
       ):
    # generate netinputs
    # feed uniform noise into the network
    nis = []
    for i in range(len(imgs)):
        if upsample_mode=="bilinear":
            # feed uniform noise into the network 
            totalupsample = 2**len(num_channels)
        elif upsample_mode=="deconv":
            # feed uniform noise into the network 
            totalupsample = 2**(len(num_channels)-1)
            #totalupsample = 2**len(num_channels)
        width = int(imgs[0].data.shape[2]/totalupsample)
        height = int(imgs[0].data.shape[3]/totalupsample)
        shape = [1 ,num_channels[0], width, height]
        print("shape: ", shape)
        net_input = Variable(torch.zeros(shape))
        net_input.data.uniform_()
        net_input.data *= 1./10
        nis.append(net_input)

    # learnable parameters are the weights
    p = [x for x in net.parameters() ]

    mse_wrt_noisy = np.zeros(num_iter)

    optimizer = torch.optim.Adam(p, lr=lr)

    mse = torch.nn.MSELoss() #.type(dtype) 

    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0

    for i in range(num_iter):
        
        def closure():
            optimizer.zero_grad()
            
            #loss = np_to_var(np.array([0.0]))
            out = net(nis[0].type(dtype))
            loss = mse(out, imgs[0].type(dtype)) 
            #for img,ni in zip(imgs,nis):
            for j in range(1,len(imgs)):
                #out = net(ni.type(dtype))
                #loss += mse(out, img.type(dtype))
                out = net(nis[j].type(dtype))
                loss += mse(out, imgs[j].type(dtype))
        
            #out = net(nis[0].type(dtype))
            #out2 = net(nis[1].type(dtype))
            #loss = mse(out, imgs[0].type(dtype)) + mse(out2, imgs[1].type(dtype))
        
            loss.backward()
            mse_wrt_noisy[i] = loss.data.cpu().numpy()
            
            if i % 10 == 0:
                print ('Iteration %05d    Train loss %f' % (i, loss.data), '\r', end='')
            return loss
        
        loss = optimizer.step(closure)
            
        if find_best:
            # if training loss improves by at least one percent, we found a new best net
            if best_mse > 1.005*loss.data:
                best_mse = loss.data
                best_net = copy.deepcopy(net)
                       
    if find_best:
        net = best_net
    return mse_wrt_noisy, nis, net        
        
      
