import copy
import torch
from utils.transform import fft_2d, ifft_2d, reshape_complex_vals_to_adj_channels, \
                            reshape_adj_channels_to_complex_vals

dtype = torch.cuda.FloatTensor


def fit(ksp_masked, net, net_input, mask, mask2=None,
        num_iter=10000, lr=0.01, dtype=torch.cuda.FloatTensor, 
        LAMBDA_TV=1e-8):
    ''' fit a network to masked k-space measurement
        args:
            ksp_masked: masked k-space of a single slice. torch variable [1,C,H,W]
            net: original network with randomly initiated weights
            net_input: randomly generated + scaled network input
            mask: 2D mask for undersampling the ksp
            mask2: 2D mask for echo2, if applying dual mask
            num_iter: number of iterations to optimize network
            lr: learning rate
        returns:
            net: the best network, whose output would be in image space
    '''            

    # initialize variables
    net_input = net_input.type(dtype)
    best_net = copy.deepcopy(net)
    best_mse = 10000.0
    #mse_wrt_ksp, mse_wrt_img = np.zeros(num_iter), np.zeros(num_iter)
    
    p = [x for x in net.parameters()]
    optimizer = torch.optim.Adam(p, lr=lr,weight_decay=0)
    mse = torch.nn.MSELoss()

    img_masked = ifft_2d(ksp_masked)

    # convert complex [nc,x,y] --> real [2*nc,x,y] to match w net output
    ksp_masked = reshape_complex_vals_to_adj_channels(ksp_masked).cuda()
    img_masked = reshape_complex_vals_to_adj_channels(img_masked)[None,:].cuda()
    mask = mask.cuda()
    if mask2 != None:
        mask2 = mask2.cuda()

    for i in range(num_iter):
        def closure(): # execute this for each iteration (gradient step)

            optimizer.zero_grad()

            out = net(net_input) # out is in img space

            out_img_masked = forwardm(out, mask, mask2) # img-->ksp, mask, convert to img
            
            loss_img = mse(out_img_masked, img_masked)

            loss_tv = (torch.sum(torch.abs(out_img_masked[:,:,:,:-1] - \
                                           out_img_masked[:,:,:,1:])) \
                     + torch.sum(torch.abs(out_img_masked[:,:,:-1,:] - \
                                           out_img_masked[:,:,1:,:])))
            loss_total = loss_img + LAMBDA_TV * loss_tv
            
            loss_total.backward(retain_graph=False)

            return loss_total

        loss = optimizer.step(closure)

        # at each iteration, check if loss improves by 1%. if so, a new best net
        loss_val = loss.data
        if best_mse > 1.005*loss_val:
            best_mse = loss_val
            best_net = copy.deepcopy(net)
   
    return best_net#, mse_wrt_ksp, mse_wrt_img

def forwardm(img, mask, mask2=None):
    ''' convert img --> ksp (must be complex for fft), apply mask
        convert back to img. input dim [2*nc,x,y], output dim [1,2*nc,x,y] 
        
        if adj (real-valued) channels:
            we have 2*nc, [re(e1) | re(e2) | im(e1) | im(e2)] 
        elif complex channels:
            we have nc, [re+im(e1) | re+im(e2)] '''

    img = reshape_adj_channels_to_complex_vals(img[0])
    ksp = fft_2d(img).cuda()
    
    if mask2==None: 
        ksp_masked_ = ksp * mask
    else: # apply dual masks, i.e. mask to e1, e2 separately
        assert ksp.shape == (16, 512, 160)
        ksp_m_1 = ksp[:8] * mask
        ksp_m_2 = ksp[8:] * mask2
        ksp_masked_ = torch.cat((ksp_m_1, ksp_m_2), 0)

    img_masked_ = ifft_2d(ksp_masked_)

    return reshape_complex_vals_to_adj_channels(img_masked_)[None, :]

def fit_many_heads(ksp_masked, net1, net_input1, net2, net_input2,
        mask, mask2=None, num_iter=10000, lr=0.01, 
        dtype=torch.cuda.FloatTensor, LAMBDA_TV=1e-8):
    ''' fit a network to masked k-space measurement
        args:
            ksp_masked: masked k-space of a single slice. torch variable [1,C,H,W]
            net: original network with randomly initiated weights
            net_input: randomly generated + scaled network input
            mask: 2D mask for undersampling the ksp
            mask2: 2D mask for echo2, if applying dual mask
            num_iter: number of iterations to optimize network
            lr: learning rate
        returns:
            net: the best network, whose output would be in image space
    '''

    # initialize variables
    net_input1 = net_input1.type(dtype)
    net_input2 = net_input2.type(dtype)
    best_net1 = copy.deepcopy(net1)
    best_net2 = copy.deepcopy(net2)
    best_mse = 10000.0
    #mse_wrt_ksp, mse_wrt_img = np.zeros(num_iter), np.zeros(num_iter)

    #p = [x for x in net.parameters()]
    p = [x for x in net1.parameters()] + [y for y in net2.parameters()]
    optimizer = torch.optim.Adam(p, lr=lr,weight_decay=0)
    mse = torch.nn.MSELoss()

    img_masked = ifft_2d(ksp_masked)

    # convert complex [nc,x,y] --> real [2*nc,x,y] to match w net output
    ksp_masked = reshape_complex_vals_to_adj_channels(ksp_masked).cuda()
    img_masked = reshape_complex_vals_to_adj_channels(img_masked)[None,:].cuda()
    mask = mask.cuda()
    if mask2 != None:
        mask2 = mask2.cuda()

    for i in range(num_iter):
        def closure(): # execute this for each iteration (gradient step)

            optimizer.zero_grad()

            out1 = net1(net_input1) # out is in img space
            out2 = net2(net_input2) 

            out_img_masked1 = forwardm(out1, mask, mask2) # img-->ksp, mask, convert to img
            out_img_masked2 = forwardm(out2, mask, mask2) 

            out_img_masked = torch.mean(torch.stack([out_img_masked1,
                                                     out_img_masked2]), dim=0)

            loss_img = mse(out_img_masked, img_masked)
            loss_tv = (torch.sum(torch.abs(out_img_masked[:,:,:,:-1] - \
                                           out_img_masked[:,:,:,1:])) \
                     + torch.sum(torch.abs(out_img_masked[:,:,:-1,:] - \
                                           out_img_masked[:,:,1:,:])))
            loss_total = loss_img + LAMBDA_TV * loss_tv

            loss_total.backward(retain_graph=False)

            return loss_total

        loss = optimizer.step(closure)

        # at each iteration, check if loss improves by 1%. if so, a new best net
        loss_val = loss.data
        if best_mse > 1.005*loss_val:
            best_mse = loss_val
            best_net1 = copy.deepcopy(net1)
            best_net2 = copy.deepcopy(net2)

    return best_net1, best_net2
