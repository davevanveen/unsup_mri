''' Various fucntions for data transformations '''

import os, sys
import numpy as np
import torch

def np_to_tt(arr):
    ''' convert numpy array to torch tensor
        if arr is complex, real/imag parts stacked on last dimn '''
    if np.iscomplexobj(arr):
        arr = np.stack((arr.real, arr.imag), axis=-1)
    return torch.from_numpy(arr)

def var_to_np(arr):
    ''' converts torch.Variable to numpy array
        shape [1, C, W, H] --> [C, W, H] '''
    return arr.data.cpu().numpy()[0]

def np_to_var(arr):
    ''' converts image in numpy.array to torch.Variable.
    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return Variable(np_to_tt(arr)[None, :])

def ifft_2d(arr):
    ''' apply centered 2D inverse fast fourier transform
    
        args:
            arr (torch.Tensor): complex valued input arr containing at least 3 dimns: dimns
            -3 & -2 are spatial dimns and dimn -1 has size 2. all other dimns are
            assumed to be batch dimns.
        returns: ifft of input (torch.Tensor)
    '''
    assert arr.size(-1) == 2
    arr = ifftshift(arr, dim=(-3, -2))
    arr = torch.ifft(arr, 2, normalized=True)
    arr = fftshift(arr, dim=(-3, -2))
    return arr

def apply_mask(arr, mask=None, mask_func=None, seed=None):
    ''' 
    subsample given k-space by multiplying with a mask.
    if no mask entered as input, generate via mask_func

    args:
        arr (torch.tensor): input k-space data of at least 3 dimns, where
            dimns -3 and -2 are the spatial dimns, and the final dimn has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int, optional): seed for the random number generator.
    returns:
        masked data (torch.tensor): subsampled k-space data
        mask (torch.tensor): the generated mask
    '''
    
    if mask is None:
        print('Want to generate new mask here')
        sys.exit()
        #shape = np.array(arr.shape)
        #shape[:-3] = 1
        #mask = mask_func(shape, seed)
    
    if arr.shape[2] != mask.shape[2]: # added to avoid dim error
        mask = mask[:, :, :arr.shape[2], :]

    return arr*mask#, mask

###################################################################
### helper functions for use within script, i.e. not for export ###

def roll(x, shift, dim):
    ''' similar to np.roll but applies to PyTorch Tensors '''
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)

def fftshift(x, dim=None):
    ''' similar to np.fft.fftshift but applies to PyTorch Tensors '''
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    ''' Similar to np.fft.ifftshift but applies to PyTorch Tensors '''
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)
