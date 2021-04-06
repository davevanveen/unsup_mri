import os, sys
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import torch
import h5py

from include.subsample import MaskFunc

def get_file_list(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files.sort()
    return files

def get_mtr_ids(file_list):
    ''' given list of files 
        return list of unique XXX entries in MTR_XXX_... ''' 
    mtr_ids = list(set([s.split('MTR_')[1][:3] for s in file_list]))
    mtr_ids.sort()    
    return mtr_ids

def get_mtr_ids_and(path_1, path_2):
    ''' given two paths, return list of matching filenames in each directory '''
    
    # get list of mtr ids in each directory
    mtr_ids_1 = get_mtr_ids(get_file_list(path_1))
    mtr_ids_2 = get_mtr_ids(get_file_list(path_2))
    
    mtr_ids_and = [a and b for a, b in zip(mtr_ids_1, mtr_ids_2)]
    mtr_ids_and.sort()
    
    if len(mtr_ids_and) == 0:
        raise ValueError('no overlapping files between paths')
        
    return mtr_ids_and

def load_imgs(mtr_id_list, path):
    ''' given list of mtr_ids and a directory 
        return single array w all imgs '''
    
    # indicator string if loading gt
    gt_str = '_gt' if '/gt/' in path else '' 

    num_samps = len(mtr_id_list)
    num_echos, num_y, num_z = 2, 512, 160
    arr = np.empty((num_samps, num_echos, num_y, num_z))
        
    for idx_s, mtr_id in enumerate(mtr_id_list):
        
        arr[idx_s, 0] = np.load('{}MTR_{}_e1{}.npy'.format(path, mtr_id, gt_str))
        arr[idx_s, 1] = np.load('{}MTR_{}_e2{}.npy'.format(path, mtr_id, gt_str))
        
    return arr

def get_mask(ksp_orig, center_fractions=[0.07], accelerations=[4]):
    ''' simplified version of get_masks() in utils.helpers -- return only a 1d mask in torch tensor '''

    mask_func = MaskFunc(center_fractions=center_fractions, \
                             accelerations=accelerations)

    # note: had to swap dims to be compatible w facebook's MaskFunc class
    mask_shape = (1, ksp_orig.shape[2], ksp_orig.shape[1])

    mask = mask_func(mask_shape, seed=0)

    return mask[0,:,0].type(torch.uint8)

def num_params(net):
    ''' given network, return total number of params '''
    return sum([np.prod(list(p.size())) for p in net.parameters()]);

def load_h5_fastmri(file_id, slice_idx=None):
    ''' given file_id, return the h5 file and central slice '''

    path_in = '/bmrNAS/people/dvv/multicoil_val/'
    filename = '{}file{}.h5'.format(path_in, file_id)
    
    f = h5py.File(filename, 'r')
    #print('file_id {} w ksp shape (num_slices, num_coils, x, y): {}'.format( \
    #                                            file_id, f['kspace'].shape))
    if not slice_idx:
        slice_idx = f['kspace'].shape[0] // 2
    slice_ksp = f['kspace'][slice_idx]

    return f, slice_ksp

def load_qdess(file_id, idx_kx=None):
    ''' load qdess w shortcut if pre-saved npy slice exists 
        default idx_kx is central in kx (axial) b/c we undersample in (ky,kz)'''

    file_in = '/bmrNAS/people/dvv/in_qdess/central_slice_in_kx/MTR_{}.npy'.format(file_id)
    
    if os.path.exists(file_in) and not idx_kx:
        return torch.from_numpy(np.load(file_in))
    
    else:
        
        ksp = load_h5_qdess(file_id)

        if idx_kx == None:
            idx_kx = ksp.shape[0] // 2
       
        # reshape, concat echo1 + echo2
        ksp_echo1 = ksp[:,:,:,0,:].permute(3,0,1,2)[:, idx_kx, :, :]
        ksp_echo2 = ksp[:,:,:,1,:].permute(3,0,1,2)[:, idx_kx, :, :]
        ksp_orig = torch.cat((ksp_echo1, ksp_echo2), 0)

        return ksp_orig

def load_h5_qdess(file_id):
    ''' given file_id, return the h5 file '''

    path_in = '/bmrNAS/people/arjun/data/qdess_knee_2020/files_recon_calib-16/'
    filename = '{}MTR_{}.h5'.format(path_in, file_id)

    f = h5py.File(filename, 'r')
    try:
        ksp = torch.from_numpy(f['kspace'][()])
    except KeyError:
        print('No kspace in file {} w keys {}'.format(fn, f.keys()))
    f.close()

    return ksp
    
### TODO: delete below code
############################################################
### BELOW CODE is to avoid running redundant expmts ########

#def check_alpha(DC_STEP, ALPHA):
#    ''' make sure specs for DC_STEP, ALPHA dont conflict '''
#
#    if not DC_STEP:
#        return 0
#    else:
#        if ALPHA == None:
#            raise ValueError('must enter value for ALPHA')
#        else:
#            return ALPHA
#
#def get_path_out(file_id, NUM_ITER, DC_STEP, ALPHA):
#    ''' given experiment specs, get the full path for output file '''
#
#    path_out = '/bmrNAS/people/dvv/out/'
#    ALPHA = check_alpha(DC_STEP, ALPHA)
#    fn_out = '{}_iter{}_alpha{}.npy'.format(file_id, NUM_ITER, ALPHA)
#
#    return path_out + fn_out
#
#def get_path_loss_curve(file_id, NUM_ITER, DC_STEP, ALPHA):
#    ''' given experiment specs, get full path for loss curve '''
#
#    path_out = '/bmrNAS/people/dvv/out/loss_curve/'
#    ALPHA = check_alpha(DC_STEP, ALPHA)
#
#    suffix = '{}_iter{}_alpha{}.npy'.format(file_id, NUM_ITER, ALPHA)
#    fn_loss_ksp = '{}mse_wrt_ksp_{}'.format(path_out, suffix)
#    fn_loss_img = '{}mse_wrt_img_{}'.format(path_out, suffix)
#
#    return fn_loss_ksp, fn_loss_img
#
#def save_output(img_dc, loss_ksp, loss_img, \
#                file_id, NUM_ITER, DC_STEP, ALPHA):
#    ''' given output img_dc and expmt params
#        save if it doesnt already exist '''
#    full_path = get_path_out(file_id, NUM_ITER, DC_STEP, ALPHA)
#    np.save(full_path, img_dc)
#    print('generated {}'.format(full_path))
#
#    fn_loss_ksp, fn_loss_img = get_path_loss_curve(\
#                                file_id, NUM_ITER, DC_STEP, ALPHA)
#    np.save(fn_loss_ksp, loss_ksp)
#    np.save(fn_loss_img, loss_img)
#
#    return
#
#def expmt_already_generated(file_id, NUM_ITER, DC_STEP, ALPHA):
#    ''' given experiment specs, check to see if file already exists
#        if so, dont re-run the experiment '''
#
#    full_path = get_path_out(file_id, NUM_ITER, DC_STEP, ALPHA)
#    if os.path.exists(full_path):
#        print('{} already generated'.format(full_path))
#        return True
#    else:
#        return False
#
#def load_output(file_id, NUM_ITER, DC_STEP, ALPHA):
#    ''' given expmt specs, load previously generated output file '''
#
#    # get reconstructed img
#    path_out = get_path_out(file_id, NUM_ITER, DC_STEP, ALPHA)
#    
#    # get loss curvers
#    path_loss_ksp, path_loss_img = get_path_loss_curve(\
#                            file_id, NUM_ITER, DC_STEP, ALPHA)
#
#    return np.load(path_out), 0, 0 #np.load(path_loss_ksp), np.load(path_loss_img)
