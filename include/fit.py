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

#dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor



# original code had exp_lr_scheduler() to decay lr by 0.1 every lr_decay_epoch epochs

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

def fit(ksp_masked, img_masked, net, net_input, mask, 
        num_iter=5000, lr=0.01, apply_f=forwardm, img_ls=None, dtype=torch.cuda.FloatTensor):
    ''' fit a network to masked k-space measurement
        args:
            ksp_masked: masked k-space of a single slice. torch variable [1,C,H,W]
            img_masked: ifft(ksp_masked). torch variable [1,C,H,W]
            net: original network with randomly initiated weights
            net_input: randomly generated + scaled network input
            mask: 2D mask for undersampling the ksp
            num_iter: number of iterations to optimize network
            lr: learning rate
            apply_f: function used to convert img to ksp, apply mask, and return masked ksp
            img_ls: least-squares image, i.e. ifft(ksp_full)
                    only needed to compute ssim, psnr over each iteration
        returns:
            net: the best network, whose output would be in image space
            mse_wrt_ksp: mse(ksp_masked, fft(out)*mask) over each iteration
            mse_wrt_img: mse(img_masked, out) over each iteration
    '''

    ### initialize variables ############### 
    # note: original code initialized out_grads, out_weights + gave opt_input option
    
    if net_input is None: # need to generate net_input via upsampling, rehaping, scaling
        raise NotImplementedError('incorporate original code here')
    net_input = net_input.type(dtype)
    noise = net_input.data.clone()

    p = [x for x in net.parameters()]
    out_imgs = np.zeros((1,1))
    
    best_net = copy.deepcopy(net)
    best_mse = 10000.0

    mse_wrt_ksp, mse_wrt_img = np.zeros(num_iter), np.zeros(num_iter)
    psnr_list, ssim_list, norm_ratio = [], [], []
    ########################################

    print("optimize with adam", lr)
    optimizer = torch.optim.Adam(p, lr=lr,weight_decay=0)
    mse = torch.nn.MSELoss()

    if img_ls is not None: 
        raise NotImplementedError('see original code to compute ssim, psnr across ' \
                                    'iterations w unmasked least-squares recon')

    #torch.set_default_dtype(torch.float16)

    for i in range(num_iter):
        def closure(): # execute this for each iteration (gradient step)
            ''' original code...
                - adjusted scaling of input
                - had try/except statement for scale_out factor when computing out and out2
                - applied mask when computing loss = mse( , )
                - provided option for showing imgs, plotting stuff, and outputting weights
                - if opt_input=True, would save best_net_input and every 100 iterations compute:
                                     loss_img when out = net(best_net_input) vs when
                                                   out = net(net_input_saved), i.e. original'''
            optimizer.zero_grad()
            out = net(net_input) # out is in image space

            # loss w.r.t. our masked k-space measurements
            # apply_f() = forwardm() converts input image to k-space, applies mask, and returns the masked k-space... confusing
            # TODO: make this apply_f call less confusing. compare forwardm to utils.transform.apply_mask()
            # make apply_f().half() if want to do with half precision i.e. torch.cuda.HalfTensor
            loss_ksp = mse(apply_f(out, mask), ksp_masked)
            # TODO: understand why we backprop on loss_ksp and not loss_img
            loss_ksp.backward(retain_graph=False)
            
            mse_wrt_ksp[i] = loss_ksp.data.cpu().numpy()

            # loss in image space
            loss_img = mse(out, img_masked.type(dtype) )
            mse_wrt_img[i] = loss_img.data.cpu().numpy()

            if i % 100 == 0:
                print ('Iteration %05d  ksp (train) loss %f  img loss %f' \
                        % (i, loss_ksp,loss_img), '\r', end='')
            
            return loss_ksp   
 
        # during forward/backward steps, use half precision
        # during update step, convert the weights to single precision
        # OR multiply by scaling factor S, then 1/S
        # OR by using autocast, which is not available in my version of torch
        loss = optimizer.step(closure)

        # at each iteration, check if loss improves by 1%. if so, a new best net
        loss_val = loss.data
        if best_mse > 1.005*loss_val:
            best_mse = loss_val
            best_net = copy.deepcopy(net)
  
    net = best_net
    
    # orig code had return options for output_gradients, output_weights, output_images,
    #                                  ssim_list, psnr_list, norm_ratio, best_net_input
    return net, mse_wrt_ksp, mse_wrt_img


        
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
        
      
