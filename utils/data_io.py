import os, sys
import numpy as np
import h5py

def load_h5(file_id):
    ''' given file_id, return the h5 file and central slice '''

    filename = '/bmrNAS/people/dvv/multicoil_val/file{}.h5'.format(file_id)
    f = h5py.File(filename, 'r')
    #print('file_id {} w ksp shape (num_slices, num_coils, x, y): {}'.format( \
    #                                            file_id, f['kspace'].shape))

    slice_idx = f['kspace'].shape[0] // 2
    slice_ksp = f['kspace'][slice_idx]

    return f, slice_ksp

############################################################
### BELOW CODE is to avoid running redundant expmts ########

def check_alpha(DC_STEP, ALPHA):
    ''' make sure specs for DC_STEP, ALPHA dont conflict '''

    if not DC_STEP:
        return 0
    else:
        if ALPHA == None:
            raise ValueError('must enter value for ALPHA')
        else:
            return ALPHA

def get_path_out(file_id, NUM_ITER, DC_STEP, ALPHA):
    ''' given experiment specs, get the full path for output file '''

    path_out = '/bmrNAS/people/dvv/out/'
    ALPHA = check_alpha(DC_STEP, ALPHA)
    fn_out = '{}_iter{}_alpha{}.npy'.format(file_id, NUM_ITER, ALPHA)

    return path_out + fn_out

def get_path_loss_curve(file_id, NUM_ITER, DC_STEP, ALPHA):
    ''' given experiment specs, get full path for loss curve '''

    path_out = '/bmrNAS/people/dvv/out/loss_curve/'
    ALPHA = check_alpha(DC_STEP, ALPHA)

    suffix = '{}_iter{}_alpha{}.npy'.format(file_id, NUM_ITER, ALPHA)
    fn_loss_ksp = '{}mse_wrt_ksp_{}'.format(path_out, suffix)
    fn_loss_img = '{}mse_wrt_img_{}'.format(path_out, suffix)

    return fn_loss_ksp, fn_loss_img

def save_output(img_dc, loss_ksp, loss_img, \
                file_id, NUM_ITER, DC_STEP, ALPHA):
    ''' given output img_dc and expmt params
        save if it doesnt already exist '''
    full_path = get_path_out(file_id, NUM_ITER, DC_STEP, ALPHA)
    np.save(full_path, img_dc)
    print('generated {}'.format(full_path))

    fn_loss_ksp, fn_loss_img = get_path_loss_curve(\
                                file_id, NUM_ITER, DC_STEP, ALPHA)
    np.save(fn_loss_ksp, loss_ksp)
    np.save(fn_loss_img, loss_img)

    return

def expmt_already_generated(file_id, NUM_ITER, DC_STEP, ALPHA):
    ''' given experiment specs, check to see if file already exists
        if so, dont re-run the experiment '''

    full_path = get_path_out(file_id, NUM_ITER, DC_STEP, ALPHA)
    if os.path.exists(full_path):
        print('{} already generated'.format(full_path))
        return True
    else:
        return False

def load_output(file_id, NUM_ITER, DC_STEP, ALPHA):
    ''' given expmt specs, load previously generated output file '''

    # get reconstructed img
    path_out = get_path_out(file_id, NUM_ITER, DC_STEP, ALPHA)
    
    # get loss curvers
    path_loss_ksp, path_loss_img = get_path_loss_curve(\
                            file_id, NUM_ITER, DC_STEP, ALPHA)

    return np.load(path_out), 0, 0 #np.load(path_loss_ksp), np.load(path_loss_img)
