import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL
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
                 upsample_mode='nearest', kernel_size=3, bias=False):
        super(Conv_Model, self).__init__()

        self.num_layers = num_layers
        self.num_channels = num_channels
        self.out_depth = out_depth
        self.hidden_size = hidden_size
        self.upsample_mode = upsample_mode
        self.kernel_size = 3
        self.bias = False

        net = nn.Sequential()
        
        # first (num_layers-1) upsampling layers
        for i in range(num_layers-1):
            net.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))#,align_corners=True))
            net.add(nn.Conv2d(in_channels=num_channels, out_channels=num_channels, 
                             kernel_size=3, stride=1, padding=1, bias=bias))
            net.add(nn.ReLU())
            net.add(nn.BatchNorm2d(num_channels, affine=True))
        
        # final layer: convert number of channels
        net.add(nn.Conv2d(in_channels=num_channels, out_channels=num_channels, \
                           kernel_size=3, stride=1, padding=1, bias=bias))
        net.add(nn.ReLU())
        net.add(nn.BatchNorm2d(num_channels, affine=True))
        net.add(nn.Conv2d(in_channels=num_channels, out_channels=out_depth, \
                           kernel_size=1, stride=1, padding=0, bias=bias))
        
        self.net = net

    def forward(self, x):
        # to return feat_maps, see Conv_Model_Old() at bottom of file
        return self.net(x)

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
    hidden_size = get_hidden_size(in_size, out_size, num_layers) # list of intermed layer sizes

    torch.manual_seed(0)

    net = Conv_Model(num_layers, num_channels, out_depth, hidden_size).type(dtype)

#     print('# parameters of ConvDecoder:',num_params(net))

    net_input = get_net_input(num_channels, in_size)
    
    # create scaled ksp to be compatible w network magnitude
    scale_factor = get_scale_factor(net, net_input, ksp_orig)
    ksp_orig_ = ksp_orig * scale_factor

    return net, net_input, ksp_orig_

###### below contains helper functions for init_convdecoder() ######
def get_hidden_size(in_size, out_size, num_layers):
    ''' determine how to scale the network based on specified input size and output size
        where output hidden_size is size of each hidden layer of network
        e.g. input [8,4] and output [640,368] would yield hidden_size of:
            [(15, 8), (28, 15), (53, 28), (98, 53), (183, 102), (343, 193), (640, 368)] '''

    # scaling factor layer-to-layer in x and y direction e.g. (scale_x, scale_y) = (1.87, 1.91)
    scale_x, scale_y = (out_size[0]/in_size[0])**(1./(num_layers-1)), \
                       (out_size[1]/in_size[1])**(1./(num_layers-1))
   
    # list of sizes for intermediate layers in [x,y] 
    hidden_size = [(int(np.ceil(scale_x**n * in_size[0])),
                    int(np.ceil(scale_y**n * in_size[1]))) for n in range(1, (num_layers-1))] + [out_size]

    return hidden_size

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
    net_output = net(net_input.type(dtype))
    net_output = net_output[0] if type(net_output) is tuple else net_output
    out = torch.from_numpy(net_output.data.cpu().numpy()[0])
    out = reshape_adj_channels_to_complex_vals(out)
    out_img = root_sum_squares(out)

    # get img of input sample
    orig = ifft_2d(ksp_orig)
    orig_img = root_sum_squares(orig)

    return torch.linalg.norm(out_img) / torch.linalg.norm(orig_img)

#class Conv_Model_Old(nn.Module):
#    def __init__(self, num_layers, num_channels, out_depth, hidden_size,
#                 upsample_mode='nearest', kernel_size=3, bias=False):
#
#        super(Conv_Model_Copied_Layers, self).__init__()
#        self.num_layers = num_layers
#        self.hidden_size = hidden_size
#        self.upsample_mode = upsample_mode
#
#        # define layer types
#        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size, \
#                         stride=1, padding=1, bias=bias)
#        self.bn = nn.BatchNorm2d(num_channels, affine=True)
#        self.collapse_channels = nn.Conv2d(num_channels, out_depth, 1, 1, padding=0, bias=bias)
#    
#        self.upsamp_list = []
#
#        for size in hidden_size:
#            self.upsamp_list.append(nn.Upsample(size=size, mode=upsample_mode))
#
#    def forward(self, x):
#        ''' run input thru net1 (convdecoder) then net2 (converts number of channels)
#        provide options for skip connections (default False) and scaling factors (default 1) '''
#        
#        feat_maps = []
#        
#        for upsamp in self.upsamp_list:
#            x = self.bn(F.relu(self.conv(upsamp(x))))
#            feat_maps.append(self.collapse_channels(x)[0])
#        
#        # last layer: collapse channels, don't upsample
#        x = self.collapse_channels(self.bn(F.relu(self.conv(x))))
#
#        return x, feat_maps
