from torch.autograd import Variable
import torch
import torch.optim
import copy
import numpy as np
from scipy.linalg import hadamard
from skimage.metrics import structural_similarity as ssim

from .helpers import *
from .mri_helpers import * #forwardm, 
from .transforms import *

dtype = torch.cuda.FloatTensor
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

def fit(net,
        img_noisy_var,
        img_clean_var,
        num_channels,
        net_input, # default None but arg always specified in ipynb
        mask, # default None but arg always specified as mask2d in ipynb
        num_iter = 5000,
        LR = 0.01, # arg sometimes specified as 0.008 in ipynb
        apply_f = forwardm, # default None but arg always specified as forwardm in ipynb
        lsimg = None, # default None but arg always specified in ipynb
        find_best=False, # default False but arg always specified to True in ipynb
        scale_out=1, # default 1, arg sometimes specified as 1 in ipynb
       ):

    ### initialize variables ############### 
    # note: original code initialied out_grads, out_weights + gave option to optimize over input
    
    if net_input is None: # need to generate net_input via upsampling, rehaping, scaling
        print('incorporate original code here')
        sys.exit()
    net_input = net_input.type(dtype)
    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()

    p = [x for x in net.parameters()]
    out_imgs = np.zeros((1,1))
    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0

    mse_wrt_noisy, mse_wrt_truth = np.zeros(num_iter), np.zeros(num_iter)
    psnr_list, ssim_list, norm_ratio = [], [], []
    ########################################

    print("optimize with adam", LR)
    optimizer = torch.optim.Adam(p, lr=LR,weight_decay=0)
    mse = torch.nn.MSELoss()
    
    for i in range(num_iter):
        
        def closure():
            ''' original code adjusted scaling of input
                              had try/except statement for scale_out factor when computing out and out2
                              applied mask when computing loss = mse( , )
                              provided option for showing images, plotting stuff, and outputting weights'''
            
            optimizer.zero_grad()
            out = net(net_input) 
                
            # training loss
            loss = mse(apply_f(out,mask), img_noisy_var)
            loss.backward(retain_graph=False)
            
            mse_wrt_noisy[i] = loss.data.cpu().numpy()

            # the actual loss TODO: figure out difference b/w training and "actual" loss
            true_loss = mse( Variable(out.data, requires_grad=False).type(dtype), img_clean_var.type(dtype) )
            mse_wrt_truth[i] = true_loss.data.cpu().numpy()
            
            if i % 100 == 0:
                if lsimg is not None: # compute ssim and psnr
                    
                    orig = crop_center2(root_sum_of_squares2(var_to_np(lsimg)), 320, 320) # least sq recon
                    out_chs = out.data.cpu().numpy()[0]
                    out_imgs = channels2imgs(out_chs)
                    rec = crop_center2(root_sum_of_squares2(out_imgs), 320, 320) # deep decoder recon

                    ssim_list.append(ssim(orig, rec, data_range=orig.max()))
                    psnr_list.append(psnr(orig, rec, np.max(orig)))

                    norm_out_img = np.linalg.norm(root_sum_of_squares2(out_imgs))
                    norm_lsimg = np.linalg.norm(root_sum_of_squares2(var_to_np(lsimg)))
                    norm_ratio.append(norm_out_img / norm_lsimg)
                
                trn_loss = loss.data
                true_loss = true_loss.data
                out2 = net(Variable(net_input_saved).type(dtype))
                loss2 = mse(out2, img_clean_var).data
                print ('Iteration %05d    Train loss %f  Actual loss %f Actual loss orig %f' % (i, trn_loss,true_loss,loss2), '\r', end='')
            
            return loss   
        
        loss = optimizer.step(closure)
            
        if find_best:
            # if training loss improves by at least one percent, we found a new best net
            lossval = loss.data
            if best_mse > 1.005*lossval:
                best_mse = lossval
                best_net = copy.deepcopy(net)
                #if opt_input:
                #    best_ni = net_input.data.clone()
                #else:
                best_ni = net_input_saved.clone()
       
    if find_best:
        net = best_net
        net_input_saved = best_ni
    
    # original code had different return options for output_gradients, output_weights, output_images, etc.
    return ssim_list, psnr_list, norm_ratio, mse_wrt_noisy, mse_wrt_truth, net_input_saved, net       
        
        
def fit_multiple(net,
        imgs, # list of images [ [1, color channels, W, H] ] 
        num_channels,
        num_iter = 5000,
        LR = 0.01,
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

    optimizer = torch.optim.Adam(p, lr=LR)

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
        
      
