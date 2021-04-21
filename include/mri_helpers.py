import torch 
import os, sys
import numpy as np
import math
import sigpy

path_m = '/home/vanveen/ConvDecoder/masks/'
path_m_arj = '/bmrNAS/people/arjun/data/qdess_knee_2020/masks/'

def load_arj_mask(accel, file_id):
    ''' load masks created by arjun '''

    data = torch.load('{}poisson_{}.0x_eval.pt'.format(path_m_arj, accel))

    mask = data['masks'][data['mtr_ids'].index('MTR_{}'.format(file_id))]
    mask = torch.from_numpy(mask).type(torch.uint8)

    # zero pad mask from (512,80) to (512,160)
    assert mask.shape == (512,80)
    mask_ = torch.zeros((512,160), dtype=torch.uint8)
    mask_[:, 40:120] = mask

    return mask_

def apply_mask(ksp_orig, accel, file_id=None, arj_mask=False, custom_calib=None):
    ''' apply mask
        default (pre 20210405): 512x160 mask w 64x64 calibration region 
        new: 512x80 mask zero padded to 512x160, w a 24x24 calibration region '''

    assert ksp_orig.shape[-2:] == (512, 160)
   

    if arj_mask:
        assert file_id
        mask = load_arj_mask(accel, file_id)
    elif custom_calib:
        rr = np.random.randint(0, high=20)
        mask_fn = '{}mask_pd_{}x_calib{}_rand{}.npy'.format(path_m, accel, custom_calib, rr)
        mask = torch.from_numpy(np.load(mask_fn))
        mask = mask.type(torch.uint8)
    else:
        mask = torch.from_numpy(np.load('{}mask_poisson_disc_{}x.npy'.format(path_m, accel)))
    
    return ksp_orig * mask, mask
    
def apply_dual_mask(ksp_orig, accel):
    ''' given echo1, echo2 concatenated together 
        apply a separate mask to each '''
    
    assert ksp_orig.shape == (16, 512, 160)
    
    mask1 = torch.from_numpy(np.load('{}mask_poisson_disc_{}x_v1.npy'.format(path_m, accel)))
    mask2 = torch.from_numpy(np.load('{}mask_poisson_disc_{}x_v2.npy'.format(path_m, accel)))
    
    ksp_e1, ksp_e2 = ksp_orig[:8], ksp_orig[8:]
    
    ksp_e1_m, ksp_e2_m = ksp_e1 * mask1, ksp_e2 * mask2
    
    return torch.cat((ksp_e1_m, ksp_e2_m), 0), mask1, mask2

def generate_t2_map(echo1, echo2, hdr = None, mask = None,
                    suppress_fat: bool = False, suppress_fluid: bool = False,
                    gl_area: float = None, tg: float = None):

    """ Generate 3D t2 map
    :param suppress_fat: Suppress fat region in t2 computation (i.e. reduce noise)
    :param gl_area: GL Area - required if not provided in the dicom
    :param tg: tg value (in microseconds) - required if not provided in the dicom
    :return MedicalVolume with 3D map of t2 values
            all invalid pixels are denoted by the value 0
    """

    T2_LOWER_BOUND = 10
    T2_UPPER_BOUND = 100
    CART_ADC = 1.25*1e-9
    CART_T1 = 1.2
    BETA_VAL = 1.2
    # All timing in seconds

    # across different scans, (tr,te) values vary slightly
    # this code sets them as constant
    try:
        TR = float(hdr.RepetitionTime) * 1e-3
        TE = float(hdr.EchoTime) * 1e-3
    except:
        TR = 18.6 * 1e-3 # 20.36 * 1e-3 per dcm
        TE = 5.9 * 1e-3 
        # e1, e2 have te of 6.428 * 1e-3, 34.292 * 1e-3, respectively

    # Flip Angle (degree -> radians)
    try:
        alpha = math.radians(float(hdr.FlipAngle))
    except:
        alpha = math.radians(20) # flip angle for dess: 20

    try:
        GlArea = float(hdr['001910b6'].value)
        Tg = float(hdr['001910b6'].value) * 1e-6
    except:
        GlArea = 3132
        Tg = 1800 * 1e-6

    Gl = GlArea / (Tg * 1e6) * 100
    gamma = 4258 * 2 * math.pi  # Gamma, Rad / (G * s).
    dkL = gamma * Gl * Tg

    # Simply math
    k = math.pow((math.sin(alpha / 2)), 2) * (
            1 + math.exp(-TR / CART_T1 - TR * math.pow(dkL, 2) * CART_ADC)) / (
                1 - math.cos(alpha) * math.exp(-TR / CART_T1 - TR * math.pow(dkL, 2) * CART_ADC))

    c1 = (TR - Tg / 3) * (math.pow(dkL, 2)) * CART_ADC

    # T2 fit
    if mask is None:
        mask = np.ones(echo1.shape)

    ratio = mask * echo2 / echo1
    ratio = np.nan_to_num(ratio)

    # have to divide division into steps to avoid overflow error
    t2map = (-2000 * (TR - TE) / (np.log(abs(ratio) / k) + c1))

    t2map = np.nan_to_num(t2map)

    # Filter calculated T2 values that are below 0ms and over 100ms
    t2map[t2map <= T2_LOWER_BOUND] = T2_LOWER_BOUND
    t2map[t2map > T2_UPPER_BOUND] = T2_UPPER_BOUND

    t2map = np.around(t2map, decimals = 2)

    # Exclude fat pixels based on signal intensity
    if suppress_fat:
        t2map = t2map * (echo1 > 0.15 * np.max(echo1))

    # Create a fluid nulled image and then exclude fluid pixels
    if suppress_fluid:
        fluid_sup = echo1-BETA_VAL*echo2
        t2map = t2map * (fluid_sup > 0.15*np.max(fluid_sup))

    tmp_mean = np.nanmean(np.where(t2map!=0,t2map,np.nan))
    tmp_std  = np.nanstd(np.where(t2map!=0,t2map,np.nan))

    # Return the T2 map and tuple for non-zero mean and std of the T2 map
    return t2map, (np.around(tmp_mean, 2), np.around(tmp_std, 2))

def generate_poisson_disc(accel, img_shape=(512,80), calib=(24,24), seed=None):
    ''' create a poisson disc mask shape (512,80) w calib region
        zero pad to (512,160) to match qdess zero padding '''

    mask = sigpy.mri.samp.poisson(img_shape=img_shape, accel=accel,
                                  calib=calib, seed=seed,
                                  crop_corner=False, max_attempts=10)
    mask = abs(torch.from_numpy(mask)).type(torch.uint8)

    accel_ = (img_shape[0]*img_shape[1]) / torch.sum(mask)
    print('given accel {}, actual accel {}'.format(
        accel, np.around(accel_,4)))

    assert img_shape == (512,80)
    mask_ = torch.zeros((512,160))
    mask_[:, 40:120] = mask

    return mask_.type(torch.uint8)

#############################################################################
### OLD CODE BELOW ##########################################################

# rectangular calib region, performs worse than square
def sample_rectangular_central_region(mask):
    ''' force central rectangular region of pixels to 1
        want 4096 pixels, b/c originally did a 64 * 64 = 4096 square region
        to be rectangular, find x s.t. (160/x) * (512/x) = 4096 --> x ~= 8.9 '''

    raise NotImplementedError('must change size of calibration region')

    size_region = 4096
    num_y, num_z = mask.shape[0], mask.shape[1] # number of pixels in (y,z)
    mid_y, mid_z = num_y // 2, num_z // 2 # middle index in (y,z)
    sf = np.sqrt((num_y * num_z) / size_region) # scaling factor
    len_y, len_z = int(num_y / (2*sf)) + 1, \
                   int(num_z / (2*sf)) + 1 # length from center, hence div by 2
    mask[mid_y-len_y:mid_y+len_y, mid_z-len_z:mid_z+len_z] = 1

    return mask

# unnecessary w calib arg in poisson
def sample_square_central_region(mask, C=24):
    ''' force calibration CxC region to be 1's'''

    idx_y, idx_z = mask.shape[0] // 2, mask.shape[1] // 2
    C_ = C // 2
    mask[idx_y-C_:idx_y+C_, idx_z-C_:idx_z+C_] = 1

    return mask

