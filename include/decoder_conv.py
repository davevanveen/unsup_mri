import torch
import torch.nn as nn
import numpy as np
from copy import copy
from torch.autograd import Variable

from utils.transform import reshape_adj_channels_to_complex_vals, \
                            root_sum_squares, ifft_2d

dtype=torch.cuda.FloatTensor

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module

class Conv_Model(nn.Module):
    def __init__(self, num_layers, num_channels, out_depth, hidden_size, 
                 upsample_mode='nearest', act_fun=nn.ReLU(), bn_affine=True, 
                 bias=False, kernel_size=3):
        
        super(Conv_Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.upsample_mode = upsample_mode
        self.act_fun = act_fun
        self.layer_inds = [] # indices of layers that generate output in the sequential mode (after each batchnorm)
        self.combinations = None # holds input of the last layer which is upsampled versions of previous layers

        # define layer types
        conv_layer = nn.Conv2d(num_channels, num_channels, kernel_size, \
                         stride=1, padding=1, bias=bias)
        batch_norm = nn.BatchNorm2d(num_channels, affine=bn_affine)
        channels_to_img = nn.Conv2d(num_channels, out_depth, 1, 1, padding=0, bias=bias)


        # first (n-1) layers - each layer is upsampling, convoltuion, relu, batchnorm
        net1 = nn.Sequential()
        
        idx_layer = 1
        for i in range(num_layers-1):
            
            net1.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))
            net1.add(conv_layer)
            net1.add(act_fun)
            idx_layer += 3 # added three layers 
            net1.add(batch_norm)

            if i != num_layers - 2: # penultimate layer will be concat'ed if skip connection option is chosen
                self.layer_inds.append(idx_layer)
            idx_layer += 1

        
        # last layer, i.e. convolution, relu, batchnorm, channels_to_img
        net2 = nn.Sequential()
        net2.add(conv_layer)
        net2.add(act_fun)
        net2.add(batch_norm)
        net2.add(channels_to_img)

        self.net1 = net1 
        self.net2 = net2 
        
    def forward(self, x):
        ''' run input thru net1 (convdecoder) then net2 (converts number of channels)
        provide options for skip connections (default False) and scaling factors (default 1) '''

        out1 = self.net1(x)
        self.combinations = copy(out1)
        out2 = self.net2(out1)
        
        return out2


def convdecoder(
        in_size, #default [16,16]
        out_size, #default [256,256]
        out_depth, #default 3
        num_layers, #default 6
        num_channels, #default 64
        ):
    
    ''' determine how to scale the network based on specified input size and output size
        where output hidden_size is size of each hidden layer of network
        e.g. input [8,4] and output [640,368] would yield hidden_size of:
            [(15, 8), (28, 15), (53, 28), (98, 53), (183, 102), (343, 193), (640, 368)]
        call Conv_Model(...), defined above 
        note: original codehad option for non-linear scaling and other args '''

    # scaling factor layer-to-layer in x and y direction e.g. (scale_x, scale_y) = (1.87, 1.91)
    scale_x, scale_y = (out_size[0]/in_size[0])**(1./(num_layers-1)), \
                       (out_size[1]/in_size[1])**(1./(num_layers-1))
   
    # list of sizes for intermediate layers in [x,y] 
    hidden_size = [(int(np.ceil(scale_x**n * in_size[0])),
                    int(np.ceil(scale_y**n * in_size[1]))) for n in range(1, (num_layers-1))] + [out_size]
    #print(hidden_size)
   
    torch.manual_seed(0)
    model = Conv_Model(num_layers, num_channels, out_depth, hidden_size)

    return model

def init_convdecoder(ksp_orig, mask, \
                     in_size=[8,4], num_layers=8, num_channels=160, kernel_size=3):
    ''' wrapper function for initializing convdecoder based on input ksp_orig

        parameters:
                ksp_orig: original, unmasked k-space measurements
                mask: mask used to downsample original k-space
        return:
                net: initialized convdecoder
                net_input: random, scaled input seed
                ksp_orig: scaled version of input '''

    out_size = ksp_orig.shape[1:] # shape of (x,y) image slice, e.g. (640, 368)
    out_depth = ksp_orig.shape[0]*2 # 2*n_c, i.e. 2*15=30 if multi-coil

    net = convdecoder(in_size, out_size, out_depth, \
                      num_layers, num_channels).type(dtype)
#     print('# parameters of ConvDecoder:',num_params(net))

    net_input = get_net_input(num_channels, in_size)
    
    # create scaled ksp to be compatible w network magnitude
    scale_factor = get_scale_factor(net, net_input, ksp_orig)
    ksp_orig_ = ksp_orig * scale_factor

    return net, net_input, ksp_orig_

def get_net_input(num_channels, in_size):
    ''' return net_input, e.g. tensor w values samples uniformly on [0,1] '''
    
    shape = [1, num_channels, in_size[0], in_size[1]]
    net_input = Variable(torch.zeros(shape)).type(dtype)
    torch.manual_seed(0)
    net_input.data.uniform_()

    return net_input

def get_scale_factor(net, net_input, ksp_orig):
    ''' return scaling factor, i.e. difference in magnitudes scaling b/w:
        original image and random image of network output = net(net_input) '''

    # generate random img
    out = torch.from_numpy(net(net_input.type(dtype)).data.cpu().numpy()[0])
    out = reshape_adj_channels_to_complex_vals(out)
    out_img = root_sum_squares(out)

    # get img of input sample
    orig = ifft_2d(ksp_orig)  
    orig_img = root_sum_squares(orig)

    return torch.linalg.norm(out_img) / torch.linalg.norm(orig_img)
