''' implementation of the high-frequency error norm metric '''

import os, sys
import torch
import torch.nn as nn
import math, numbers

class HFENLoss(nn.Module): # Edge loss with pre_smooth
    """Calculates high frequency error norm (HFEN) between target and
     prediction used to quantify the quality of reconstruction of edges
     and fine features.
     Uses a rotationally symmetric LoG (Laplacian of Gaussian) filter to
     capture edges. The original filter kernel is of size 15×15 pixels,
     and has a standard deviation of 1.5 pixels.
     ks = 2 * int(truncate * sigma + 0.5) + 1, so use truncate=4.5
     HFEN is computed as the norm of the result obtained by LoG filtering the
     difference between the reconstructed and reference images.
    [1]: Ravishankar and Bresler: MR Image Reconstruction From Highly
    Undersampled k-Space Data by Dictionary Learning, 2011
        https://ieeexplore.ieee.org/document/5617283
    [2]: Han et al: Image Reconstruction Using Analysis Model Prior, 2016
        https://www.hindawi.com/journals/cmmm/2016/7571934/
    Parameters
    ----------
    img1 : torch.Tensor or torch.autograd.Variable
        Predicted image
    img2 : torch.Tensor or torch.autograd.Variable
        Target image
    norm: if true, follows [2], who define a normalized version of HFEN.
        If using RelativeL1 criterion, it's already normalized.
    """
    def __init__(self, loss_f=torch.nn.MSELoss(), kernel='log', kernel_size=15, sigma = 2.5, norm = True): #1.4 ~ 1.5
        super(HFENLoss, self).__init__()
        # can use different criteria
        self.criterion = loss_f
        self.norm = norm
        #can use different kernels like DoG instead:
        if kernel == 'dog':
            kernel = get_dog_kernel(kernel_size, sigma)
        else:
            kernel = get_log_kernel(kernel_size, sigma)
        self.filter = load_filter(kernel=kernel, kernel_size=kernel_size)

    def forward(self, img1, img2):
        self.filter.to(img1.device)
        # HFEN loss
        log1 = self.filter(img1)
        log2 = self.filter(img2)
        hfen_loss = self.criterion(log1, log2)
        if self.norm:
            hfen_loss /= img2.norm()
        return hfen_loss

def get_log_kernel_5x5():
    '''
    This is a precomputed LoG kernel that has already been convolved with
    Gaussian, for edge detection.
    http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
    http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    https://academic.mu.edu/phys/matthysd/web226/Lab02.htm
    The 2-D LoG can be approximated by a 5 by 5 convolution kernel such as:
    This is an approximate to the LoG kernel with kernel size 5 and optimal
    sigma ~6 (0.590155...). '''
    return torch.Tensor([
                [0, 0, 1, 0, 0],
                [0, 1, 2, 1, 0],
                [1, 2, -16, 2, 1],
                [0, 1, 2, 1, 0],
                [0, 0, 1, 0, 0]
            ])

#dim is the image dimension (2D=2, 3D=3, etc), but for now the final_kernel is hardcoded to 2D images
#Not sure if it would make sense in higher dimensions
#Note: Kornia suggests their laplacian kernel can also be used to generate LoG kernel:
# https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/filters/laplacian.html
def get_log_kernel2d(kernel_size=5, sigma=None, dim=2): #sigma=0.6; kernel_size=5

    #either kernel_size or sigma are required:
    if not kernel_size and sigma:
        kernel_size = get_kernel_size(sigma)
        kernel_size = [kernel_size] * dim #note: should it be [kernel_size] or [kernel_size-1]? look below
    elif kernel_size and not sigma:
        sigma = get_kernel_sigma(kernel_size)
        sigma = [sigma] * dim

    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size-1] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    grids = torch.meshgrid([torch.arange(-size//2,size//2+1,1) for size in kernel_size])

    kernel = 1
    for size, std, mgrid in zip(kernel_size, sigma, grids):
        kernel *= torch.exp(-(mgrid**2/(2.*std**2)))

    #TODO: For now hardcoded to 2 dimensions, test to make it work in any dimension
    final_kernel = (kernel) * ((grids[0]**2 + grids[1]**2) - (2*sigma[0]*sigma[1])) * (1/((2*math.pi)*(sigma[0]**2)*(sigma[1]**2)))

    #TODO: Test if normalization has to be negative (the inverted peak should not make a difference)
    final_kernel = -final_kernel / torch.sum(final_kernel)

    return final_kernel

def get_log_kernel(kernel_size: int = 5, sigma: float = None, dim: int = 2):
    '''
        Returns a Laplacian of Gaussian (LoG) kernel. If the kernel is known, use it,
        else, generate a kernel with the parameters provided (slower)
    '''
    if kernel_size ==5 and not sigma and dim == 2:
        return get_log_kernel_5x5()
    else:
        return get_log_kernel2d(kernel_size, sigma, dim)

def load_filter(kernel, kernel_size=3, in_channels=1, out_channels=1,
                stride=1, padding=True, groups=1, dim: int =2,
                requires_grad=False):
    '''
        Loads a kernel's coefficients into a Conv layer that
            can be used to convolve an image with, by default,
            for depthwise convolution
        Can use nn.Conv1d, nn.Conv2d or nn.Conv3d, depending on
            the dimension set in dim (1,2,3)
        #From Pytorch Conv2D:
            https://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#Conv2d
            When `groups == in_channels` and `out_channels == K * in_channels`,
            where `K` is a positive integer, this operation is also termed in
            literature as depthwise convolution.
             At groups= :attr:`in_channels`, each input channel is convolved with
             its own set of filters, of size:
             :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.
    '''

    '''#TODO: check if this is necessary, probably not
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
    '''

    # Reshape to 2d depthwise convolutional weight
    kernel = kernel_conv_w(kernel, in_channels)
    assert(len(kernel.shape)==4 and kernel.shape[0]==in_channels)

    if padding:
        pad = compute_padding(kernel_size)
    else:
        pad = 0

    # create filter as convolutional layer
    if dim == 1:
        conv = nn.Conv1d
    elif dim == 2:
        conv = nn.Conv2d
    elif dim == 3:
        conv = nn.Conv3d
    else:
        raise RuntimeError(
            'Only 1, 2 and 3 dimensions are supported for convolution. \
            Received {}.'.format(dim)
        )

    filter = conv(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride, padding=padding,
                        groups=groups, bias=False)
    filter.weight.data = kernel
    filter.weight.requires_grad = requires_grad
    return filter

def kernel_conv_w(kernel, channels: int =3):
    '''
        Reshape a H*W kernel to 2d depthwise convolutional
            weight (for loading in a Conv2D)
    '''

    # Dynamic window expansion. expand() does not copy memory, needs contiguous()
    kernel = kernel.expand(channels, 1, *kernel.size()).contiguous()
    return kernel

def compute_padding(kernel_size):
    '''
        Computes padding tuple. For square kernels, pad can be an
         int, else, a tuple with an element for each dimension
    '''
    # 4 or 6 ints:  (padding_left, padding_right, padding_top, padding_bottom)
    if isinstance(kernel_size, int):
        return kernel_size//2
    elif isinstance(kernel_size, list):
        computed = [k // 2 for k in kernel_size]

        out_padding = []

        for i in range(len(kernel_size)):
            computed_tmp = computed[-(i + 1)]
            # for even kernels we need to do asymetric padding
            if kernel_size[i] % 2 == 0:
                padding = computed_tmp - 1
            else:
                padding = computed_tmp
            out_padding.append(padding)
            out_padding.append(computed_tmp)
        return out_padding
