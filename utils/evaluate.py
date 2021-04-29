''' various functions for computing image quality metrics '''

import h5py
import scipy
import numpy as np
import torch
import matplotlib.pyplot as plt

from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_msssim import ms_ssim


def calc_metrics_imgs(imgs_gt, imgs_out):
    ''' given (gt, out) imgs over many qdess samples / echos
        compute metrics for each '''
    
    assert imgs_gt.shape == imgs_out.shape
    assert imgs_gt.shape[-2:] == (512, 160)
    assert imgs_gt.ndim == 4
    
    num_samps = imgs_gt.shape[0]
    num_echos = imgs_gt.shape[1]
    num_metrics = 4 # vif, msssim, ssim, psnr
    
    metrics = np.empty((num_samps, num_echos, num_metrics))
    
    for idx_s in np.arange(num_samps):
        for idx_e in np.arange(num_echos):
            
            im_gt, im_out = imgs_gt[idx_s, idx_e], imgs_out[idx_s, idx_e]
            metrics[idx_s, idx_e] = np.array(calc_metrics(im_gt, im_out))
            
    return np.around(metrics, decimals=4)

def calc_metrics(img_gt, img_out):
    ''' compute vif, mssim, ssim, and psnr of img_out using img_gt as ground-truth reference 
        note: for msssim, made a slight mod to source code in line 200 of 
              /home/vanveen/heck/lib/python3.8/site-packages/pytorch_msssim/ssim.py 
              to compute msssim over images w smallest dim >=160 '''

    img_gt, img_out = norm_imgs(img_gt, img_out)
    img_gt, img_out = np.array(img_gt), np.array(img_out)

    vif_ = vifp_mscale(img_gt, img_out, sigma_nsq=img_out.mean())
    ssim_ = ssim(img_gt, img_out)
    psnr_ = psnr(img_gt, img_out)

    img_out_ = torch.from_numpy(np.array([[img_out]]))
    img_gt_ = torch.from_numpy(np.array([[img_gt]]))
    msssim_ = float(ms_ssim(img_out_, img_gt_, data_range=img_gt_.max()))

    return vif_, msssim_, ssim_, psnr_

def norm_imgs(img_gt, img_out):
    ''' first, normalize ground-truth img_gt to be on range [0,.1] 
               note: this step has a significant effect on vif metric score 
                               has no effect on other metric scores 
                               
        second, normalize predicted img_out according to range of img_gt '''
    
    mu, sig = img_gt.mean(), img_gt.std()
    C = .1 / img_gt.max()
    img_gt = (img_gt - mu) / sig
    img_gt *= (C*sig)
    img_gt += (C*mu)
    
    img_out = (img_out - img_out.mean()) / img_out.std()
    img_out *= img_gt.std()
    img_out += img_gt.mean()
    
    return img_gt, img_out

def scale_0_1(arr):
    ''' given any array, map it to [0,1] range '''
    return (arr - arr.min()) * (1. / arr.max())

def plot_row_qdess(arr_list, title_list=None, clim_list=None):
    ''' given list of imgs, plot a single row for comparison
        e.g. arr_list=[im_gt, im_1, im_2]
             titel_list=[gt, method 1, method 2] '''

    SF = 2.56 # stretch factor
    NUM_COLS = len(arr_list)
    if not title_list:
        title_list = NUM_COLS * ['']

    fig = plt.figure(figsize=(10,10))

    for idx in range(NUM_COLS):
        ax = fig.add_subplot(1,NUM_COLS,idx+1)
        ax.imshow(arr_list[idx], cmap='gray', \
                  clim=clim_list[idx], aspect=1./SF)
        ax.set_title(title_list[idx], fontsize=20)
        ax.axis('off')

##############################################################
# below contains wrapper functions for computing quant metrics 
def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)

def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())

def ssim(gt, pred):
    ''' compute structural similarity index metric (ssim) 
        NOTE: can get higher values by using data_range=gt.max() '''
    return structural_similarity(gt, pred, multichannel=False, data_range=pred.max())

def vifp_mscale(ref, dist, sigma_nsq=1, eps=1e-10):
    ''' from https://github.com/aizvorski/video-quality/blob/master/vifp.py
        ref: reference ground-truth image
        dist: distorted image to evaluate
        sigma_nsq: ideally tune this according to input pixel values

        NOTE: order of ref, dist is important '''

    num = 0.0
    den = 0.0
    for scale in range(1, 5):
       
        N = 2**(4-scale+1) + 1
        sd = N/5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]
                
        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2
        
        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0
        
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        
        g[sigma1_sq<eps] = 0
        sv_sq[sigma1_sq<eps] = sigma2_sq[sigma1_sq<eps]
        sigma1_sq[sigma1_sq<eps] = 0
        
        g[sigma2_sq<eps] = 0
        sv_sq[sigma2_sq<eps] = 0
        
        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=eps] = eps
        
        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
        
    vifp = num/den

    return vifp

METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    VIF=vifp_mscale,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )


def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file) as target, h5py.File(
          args.predictions_path / tgt_file.name) as recons:
            if args.acquisition and args.acquisition != target.attrs['acquisition']:
                continue
            target = target[recons_key].value
            recons = recons['reconstruction'].value
            metrics.push(target, recons)
    return metrics
