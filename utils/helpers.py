''' miscellaneous data i/o and helper functions '''

import os, sys
import numpy as np
import torch
import h5py

from utils.transform import np_to_tt


def load_h5(file_id, slice_idx_from_last=None):
    ''' given file_id, return the h5 file and central slice '''

    filename = '/bmrNAS/people/dvv/multicoil_test_v2/file{}_v2.h5'.format(file_id)
    f = h5py.File(filename, 'r')
    #print('file_id {} w ksp shape (num_slices, num_coils, x, y): {}'.format( \
    #                                            file_id, f['kspace'].shape))

    if f['kspace'].shape[3] == 320:
        print('2D slice is length 320 -- may prevent masks from loading properly')

    if not slice_idx_from_last: # isolate central k-space slice
        slice_idx = f['kspace'].shape[0] // 2
    else: # isolate slice at _ distance from last
        slice_idx = f['kspace'].shape[0] - slice_idx_from_last
    slice_ksp = f['kspace'][slice_idx]

    return f, slice_ksp

def get_masks(file_h5, slice_ksp):
    ''' h5 file contains 1d binary vector to denote sampling of vertical lines 
        given this, return three different versions of masks:`
            mask: used for masking k-space as network input
            mask2d: 2D mask used to fit network 
            mask1d: 1D mask used for data consistency step '''

    try:
        mask1d = np.array([1 if e else 0 for e in file_h5["mask"]]) # load 1D binary mask
    except:
        raise NotImplementedError('Implement method for generating a new mask here')

    # zero out mask in outer regions e.g. mask and data have last dimn 368, but actual data is size 320
    # TODO: make sure this works for 320x320 slices
    idxs_zero = (mask1d.shape[-1] - 320) // 2 # e.g. zero first/last (368-320)/2=24 indices
    mask1d[:idxs_zero], mask1d[-idxs_zero:] = 0, 0

    # create 2d mask. zero pad if dimensions don't line up
    mask2d = np.repeat(mask1d[None,:], slice_ksp.shape[1], axis=0)#.astype(int)
    mask2d = np.pad(mask2d, ((0,),((slice_ksp.shape[-1]-mask2d.shape[-1])//2,)), mode='constant')

    # convert shape e.g. (368,) --> (1, 1, 368, 1)
    mask = np_to_tt(np.array([[mask2d[0][np.newaxis].T]])).type(torch.FloatTensor)
    #print('under-sampling factor:', round(len(mask1d) / sum(mask1d), 2))
    
    return mask, mask2d, mask1d

def num_params(net):
    ''' given network, return total number of params '''
    return sum([np.prod(list(p.size())) for p in net.parameters()]);
