import os, sys
from os import listdir
from os.path import isfile, join
import numpy as np
import h5py
import torch

path_in = '/bmrNAS/people/arjun/data/qdess_knee_2020/files_recon_calib-16/'
files = [f for f in listdir(path_in) if isfile(join(path_in, f))]
files.sort()
NUM_SAMPS = 25

path_out = '/bmrNAS/people/dvv/in_qdess/'

for fn in files:
        
        # load data
        f = h5py.File(path_in + fn, 'r')
        try:
            ksp = torch.from_numpy(f['kspace'][()])
        except KeyError:
            print('No kspace in file {} w keys {}'.format(fn, f.keys()))
            f.close()
            continue
        f.close()
                                                                
        print('{}: saved k-space data shape {}'.format(fn, ksp.shape))
        np.save('{}{}_kspace.npy'.format(path_out, fn.split('.h5')[0]), ksp)
