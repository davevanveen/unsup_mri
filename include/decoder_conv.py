import torch
import torch.nn as nn
import numpy as np
from copy import copy

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module

class conv_model(nn.Module):
    def __init__(self, num_layers, strides, num_channels, out_depth, hidden_size, upsample_mode, act_fun, bn_affine=True, bias=False, need_last=False, kernel_size=3):
        super(conv_model, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.upsample_mode = upsample_mode
        self.act_fun = act_fun
        self.layer_inds = [] # record index of the layers that generate output in the sequential mode (after each BatchNorm)
        self.combinations = None # this holds input of the last layer which is upsampled versions of previous layers
        #self.dtype = dtype

        cntr = 1
        #torch.set_default_tensor_type(dtype)
        net1 = nn.Sequential()
        for i in range(num_layers-1):
            
            net1.add(nn.Upsample(size=hidden_size[i], mode=upsample_mode))#,align_corners=True))
            cntr += 1
            
            conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i], padding=(kernel_size-1)//2, bias=bias)
            net1.add(conv)
            cntr += 1
            
            net1.add(act_fun)
            cntr += 1
            
            net1.add(nn.BatchNorm2d( num_channels, affine=bn_affine))
            if i != num_layers - 2: # penultimate layer will automatically be concatenated if skip connection option is chosen
                self.layer_inds.append(cntr)
            cntr += 1

        net2 = nn.Sequential()
        
        nic = num_channels
        
        if need_last: # orignal code default False, but we call it True
            net2.add( nn.Conv2d(nic, num_channels, kernel_size, strides[i], padding=(kernel_size-1)//2, bias=bias) )
            net2.add(act_fun)
            net2.add(nn.BatchNorm2d( num_channels, affine=bn_affine))
            nic = num_channels
            
        net2.add(nn.Conv2d(nic, out_depth, 1, 1, padding=0, bias=bias))
        
        self.net1 = net1 # actual convdecoder network
        self.net2 = net2 # (default seting) one-layer net converting number of channels
        
    def forward(self, x, scale_out=1):
        ''' run input thru net1 (convdecoder) then net2 (converts number of channels
        provide options for skip connections (default False) and scaling factors (default 1) '''
        out1 = self.net1(x)
        self.combinations = copy(out1)
        out2 = self.net2(out1)
        return out2*scale_out
    def up_sample(self,img):
        ''' single upsampling layer '''
        samp_block = nn.Upsample(size=self.hidden_size[-1], mode=self.upsample_mode)#,align_corners=True)
        img = samp_block(img)
        return img

def convdecoder(
        in_size, #default [16,16]
        out_size,#default [256,256]
        out_depth, #default 3
        num_layers, #default 6
        strides, #default [1]*6,
        num_channels, #default 64

        kernel_size=3,
        upsample_mode='nearest', #default 'bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_affine = True,
        nonlin_scales=False,
        bias=False,
        need_last=True, #False,
        ):
    
    ''' determine how to scale the network based on specified input size and output size
        where output hidden_size is size of each hidden layer of network
        e.g. input [8,4] and output [640,368] would yield hidden_size of:
            [(15, 8), (28, 15), (53, 28), (98, 53), (183, 102), (343, 193), (640, 368)]
        provide option for nonlinear scaling (default False) and different activation functions
        call conv_model(...), defined above 

        Note: I removed unnecessary args, e.g. skips, intermeds, pad, etc. 
              decoder_conv_old.py for original code'''

    # scaling factor layer-to-layer in x and y direction
    # e.g. (scale_x, scale_y) = (1.87, 1.91)
    scale_x,scale_y = (out_size[0]/in_size[0])**(1./(num_layers-1)), (out_size[1]/in_size[1])**(1./(num_layers-1))
    
    if nonlin_scales: # default false
        xscales = np.ceil( np.linspace(scale_x * in_size[0],out_size[0],num_layers-1) )
        yscales = np.ceil( np.linspace(scale_y * in_size[1],out_size[1],num_layers-1) )
        hidden_size = [(int(x),int(y)) for (x,y) in zip(xscales,yscales)]
    else:
        hidden_size = [(int(np.ceil(scale_x**n * in_size[0])),
                        int(np.ceil(scale_y**n * in_size[1]))) for n in range(1, (num_layers-1))] + [out_size]
    #print(hidden_size)
    
    model = conv_model(num_layers, strides, num_channels, out_depth, hidden_size,
                         upsample_mode=upsample_mode, 
                         act_fun=act_fun,
                         bn_affine=bn_affine,
                         bias=bias,
                         need_last=need_last,
                         kernel_size=kernel_size)#,
                         #dtype=dtype)
    return model
