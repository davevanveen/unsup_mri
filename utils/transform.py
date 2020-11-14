''' various functions for data transformations '''

import os, sys
import numpy as np
import torch
from torch.autograd import Variable
from torch.fft import fftn, ifftn


def reshape_adj_channels_to_complex_vals(arr):
    ''' reshape real tensor dim [2*nc,x,y] --> complex tensor dim [nc,x,y]
        assumes first nc sub-arrays are real, second nc are imag i.e. not alternating 
        inverse operation of reshape_complex_vals_to_adj_channels() '''

    assert not is_complex(arr) # input should be real-valued

    nc = int(arr.shape[0] // 2) # num_channels
    arr_out = torch.empty((nc, arr.shape[1], arr.shape[2]), dtype=torch.complex64)
    arr_out.real = arr[0:nc]
    arr_out.imag = arr[nc:2*nc]
    #for idx in range(nc):
    #    arr_out[idx].real = arr[2*idx]
    #    arr_out[idx].imag = arr[2*idx+1]
        
    return arr_out

def reshape_complex_vals_to_adj_channels(arr):
    ''' reshape complex tensor dim [nc,x,y] --> real tensor dim [2*nc,x,y]
        s.t. concat([nc,x,y] real, [nc,x,y] imag), i.e. not alternating real/imag 
        inverse operation of reshape_adj_channels_to_complex_vals() '''

    assert is_complex(arr) # input should be complex-valued
    
    #nc = int(arr.shape[0])
    #arr_out = torch.empty((2*nc, arr.shape[1], arr.shape[2]))
    #for idx in range(nc):
    #    arr_out[2*idx] = arr[idx].real
    #    arr_out[2*idx+1] = arr[idx].imag
    #
    #return arr_out
    return torch.cat([torch.real(arr), torch.imag(arr)])

# TODO: delete deprecated functions commented out below
#def recon_ksp_to_img(ksp, dim=320):
#    ''' given a 3D npy array (or torch tensor) ksp k-space e.g. shape (15,x,y)
#        (1) perform ifft to img space
#        (2) reshape/combine complex channels
#        (3) combine multiple coils via root sum of squares
#        (4) crop center portion of image according to dim
#    '''
#
#    if type(ksp).__module__ == np.__name__:
#        ksp = np_to_tt(ksp)
#
#    arr = ifft_2d(ksp).cpu().numpy() # (15,x,y,2) --> (15,x,y,2)
#    arr = reshape_complex_channels_to_be_adj(arr) # (15,x,y,2) --> (30,x,y)
#    arr = combine_complex_channels(arr) # e.g. shape (30,x,y) --> (15,x,y)
#    arr = root_sum_of_squares(arr) # e.g. (15,x,y) --> (x,y)
#    arr = crop_center(arr, dim, dim) # e.g. (x,y) --> (dim,dim)
#
#    return arr

#def np_to_tt(arr):
#    ''' convert numpy array to torch tensor
#        if arr is complex, real/imag parts stacked on last dimn '''
#    if np.iscomplexobj(arr):
#        arr = np.stack((arr.real, arr.imag), axis=-1)
#    return torch.from_numpy(arr)
#
#def var_to_np(arr):
#    ''' converts torch.Variable to numpy array
#        shape [1, C, W, H] --> [C, W, H] '''
#    return arr.data.cpu().numpy()[0]
#
#def np_to_var(arr):
#    ''' converts image in numpy.array to torch.Variable.
#    From C x W x H [0..1] to  1 x C x W x H [0..1]
#    '''
#    return Variable(np_to_tt(arr)[None, :])
#
#def split_complex_vals(ksp):
#    # TODO: implement this in torch -- much faster
#    ''' given complex npy array, split real/complex vals into two channels 
#        e.g. shape (15,x,y) --> (15,x,y,2) '''
#    return np.transpose(np.array([np.real(ksp),np.imag(ksp)]), (1, 2, 3, 0))
#
#def reshape_complex_channels_to_be_adj(arr):
#    ''' e.g. reshape numpy arr shape (15,x,y,2) --> (30,x,y) '''
#    arr_out = []
#    for a in arr:
#        arr_out += [a[:,:,0], a[:,:,1]]
#    return np.array(arr_out)

#def reshape_complex_channels_to_sep_dimn(arr):
#    ''' e.g. reshape torch tensor shape [(30,x,y)] --> [(15,x,y,2)] '''
#
#    num_slices = int(arr.shape[0]/2) # 15*2=30, i.e. real/complex separate
#
#    shape_out = (num_slices, arr.shape[1], arr.shape[2], 2)
#    arr_out = torch.zeros(shape_out)
#    for i in range(num_slices):
#        arr_out[i,:,:,0] = arr[2*i,:,:]
#        arr_out[i,:,:,1] = arr[2*i+1,:,:]
#
#    return arr_out

#def reshape_adj_channels_to_be_complex(arr):
#    ''' given 3d real tensor shape [2*nc, x, y] with alternating real/complex slices in first dimn
#        return 3d complex tensor shape [nc, x, y] '''
#    
#    assert  not is_complex(arr)
#    assert len(arr.shape) == 3 # i.e. (2*nc, x, y)
#    
#    num_slices = int(arr.shape[0]/2) # 15*2=30, i.e. real/complex separate
#    shape_out = (num_slices, arr.shape[1], arr.shape[2])
#    arr_out = torch.empty(shape_out, dtype=torch.complex64)
#    
#    for i in range(num_slices):
#        arr_out[i,:,:].real = arr[2*i,:,:] 
#        arr_out[i,:,:].imag = arr[2*i+1,:,:]
#
#    return arr_out

#def combine_complex_channels(arr):
#    ''' e.g. given npy array of shape (30,x,y)
#        combine real/complex values into a single magnitude
#        return a npy array of shape (15,x,y) '''
#    num_coils = int(arr.shape[0]/2)
#    arr_out = np.zeros((num_coils, arr.shape[1], arr.shape[2]))
#    for i in range(num_coils):
#        arr_out[i] = np.sqrt(arr[2*i]**2 + arr[2*i+1]**2)
#    return arr_out

def crop_center(arr, crop_x, crop_y):
    ''' given 2D npy array, crop center area according to given dimns cropx, cropy '''
    x0 = (arr.shape[1] // 2) - (crop_x // 2)
    y0 = (arr.shape[0] // 2) - (crop_y // 2)
    return arr[y0:y0+crop_y, x0:x0+crop_x]

# TODO: delete this - should only process w torch tensors
#def root_sum_of_squares(arr):
#    ''' given 3D npy array e.g. 2D slices from multiple coils
#        combine each slice into a single 2D array via rss '''
#    return np.sqrt(np.sum(np.square(arr), axis=0))

def ifft_2d(arr):
    ''' apply centered 2d ifft, where (-2, -1) are spatial dimensions of arr '''
    
    assert is_complex(arr) # input must be complex-valued

    dims = (-2,-1)

    arr = ifftshift(arr, dim=dims)
    # TODO: resolve! added norm='ortho' for fastmri expmt, but didn't have that arg for qdess
    arr = torch.fft.ifftn(arr, dim=dims, norm='ortho')
    arr = fftshift(arr, dim=dims)

    return arr


def fft_2d(arr):
    ''' apply centered 2d fft, where (-2, -1) are spatial dimensions of arr '''

    assert is_complex(arr) # input must be complex-valued

    dims=(-2,-1)

    arr = ifftshift(arr, dim=dims)
    # TODO: resolve! added norm='ortho' for fastmri expmt, but didn't have that arg for qdess
    arr = torch.fft.fftn(arr, dim=dims, norm='ortho')
    arr = fftshift(arr, dim=dims)

    return arr

def root_sum_squares(arr):
    ''' given 3d complex arr [nc,x,y], perform rss over magnitudes 
        return 2d arr [x,y] '''
    
    assert is_complex(arr)
    return torch.sqrt(torch.sum(torch.square(abs(arr)), axis=0))

def is_complex(arr):
    dt = arr.dtype
    return dt==torch.complex64 or dt==torch.complex128 or dt==torch.complex32

# TODO: delete this
#def apply_mask(arr, mask=None, mask_func=None, seed=None):
#    ''' 
#    subsample given k-space by multiplying with a mask.
#    if no mask entered as input, generate via mask_func
#
#    args:
#        arr (torch.tensor): input k-space data of at least 3 dimns, where
#            dimns -3 and -2 are the spatial dimns, and the final dimn has size
#            2 (for complex values).
#        mask_func (callable): A function that takes a shape (tuple of ints) and a random
#            number seed and returns a mask.
#        seed (int, optional): seed for the random number generator.
#    returns:
#        masked data (torch.tensor): subsampled k-space data
#        mask (torch.tensor): the generated mask
#    '''
#    
#    if mask is None:
#        raise NotImplementedError('Need to gnerate new mask')
#        #shape = np.array(arr.shape)
#        #shape[:-3] = 1
#        #mask = mask_func(shape, seed)
#    
#    if arr.shape[2] != mask.shape[2]: # added to avoid dim error
#        mask = mask[:, :, :arr.shape[2], :]
#
#    return arr*mask#, mask

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
