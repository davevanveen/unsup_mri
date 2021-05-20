''' various functions for data transformations '''

import torch
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
        
    return arr_out

def reshape_complex_vals_to_adj_channels(arr):
    ''' reshape complex tensor dim [nc,x,y] --> real tensor dim [2*nc,x,y]
        s.t. concat([nc,x,y] real, [nc,x,y] imag), i.e. not alternating real/imag 
        inverse operation of reshape_adj_channels_to_complex_vals() '''

    assert is_complex(arr) # input should be complex-valued
    
    return torch.cat([torch.real(arr), torch.imag(arr)])

def crop_center(arr, crop_x, crop_y):
    ''' given 2D npy array, crop center area according to given dimns cropx, cropy '''
    x0 = (arr.shape[1] // 2) - (crop_x // 2)
    y0 = (arr.shape[0] // 2) - (crop_y // 2)
    return arr[y0:y0+crop_y, x0:x0+crop_x]

def ifft_2d(arr):
    ''' apply centered 2d ifft, where (-2, -1) are spatial dimensions of arr '''
    
    assert is_complex(arr) # input must be complex-valued

    dims = (-2,-1)

    arr = ifftshift(arr, dim=dims)
    # note: added norm='ortho' for fastmri expmt, but didn't have that arg for qdess
    arr = torch.fft.ifftn(arr, dim=dims, norm='ortho')
    arr = fftshift(arr, dim=dims)

    return arr


def fft_2d(arr):
    ''' apply centered 2d fft, where (-2, -1) are spatial dimensions of arr '''

    assert is_complex(arr) # input must be complex-valued

    dims=(-2,-1)

    arr = ifftshift(arr, dim=dims)
    # note: added norm='ortho' for fastmri expmt, but didn't have that arg for qdess
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
