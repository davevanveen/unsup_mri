''' miscellaneous data i/o and helper functions '''

import os, sys
import numpy as np
import torch
import h5py

from include.subsample import MaskFunc

def load_h5(file_id):
    ''' given file_id, return the h5 file and central slice '''

    filename = '/bmrNAS/people/dvv/multicoil_val/file{}.h5'.format(file_id)
    f = h5py.File(filename, 'r')
    #print('file_id {} w ksp shape (num_slices, num_coils, x, y): {}'.format( \
    #                                            file_id, f['kspace'].shape))

    slice_idx = f['kspace'].shape[0] // 2
    slice_ksp = f['kspace'][slice_idx]

    return f, slice_ksp

def num_params(net):
    ''' given network, return total number of params '''
    return sum([np.prod(list(p.size())) for p in net.parameters()]);
