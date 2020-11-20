from test_tube import Experiment
from test_tube import HyperOptArgumentParser
import datetime
import pytz

from train import train

NUM_ITER = 2000
GPU_ID = 2 

exp_list = [-1, -2]#, -3, -4, -5]
ALPHA_FM_LIST = [10**e for e in exp_list]
ALPHA_FM_LIST = [0] + ALPHA_FM_LIST

NUM_ITER = 1
ITER_START_FM_LOSS = [0, int(0.5*NUM_ITER), int(0.8*NUM_ITER)]

WEIGHT_METHODS = ['all', 'early', 'late']
DOWNSAMP_METHODS = ['bicubic', 'bilinear', 'nearest']
NUM_TRIALS = 81

def init_parser():

    parser = HyperOptArgumentParser(strategy="random_search")

    parser.opt_list('--alpha_fm', type=float, default=0, 
                    options=ALPHA_FM_LIST, tunable=True, 
                    help='weight on feat_map loss')

    parser.opt_list('--iter_start_fm_loss', type=int, default=0,
                    options=ITER_START_FM_LOSS, tunable=True,
                    help='iteration at which to incorporate fm loss')

    parser.opt_list('--weight_method', type=str, default='all',
                    options=WEIGHT_METHODS, tunable=True,
                    help='fm loss on early, later, or all net layers')

    parser.opt_list('--downsamp_method', type=str, default='bicubic',
                    options=DOWNSAMP_METHODS, tunable=True,
                    help='interpolation method on ksp_masked')

    parser.add_argument('--num_iter', type=int, default=NUM_ITER,
                        help='number of gradient descent iterations')

    csv_fn = init_csv()
    parser.add_argument('--csv_fn', type=str, default=csv_fn,
                        help='csv filename for results')

    parser.add_argument('--gpu_id', type=int, default=GPU_ID,
                        help='gpu to use for training')

    hparams = parser.parse_args()

    return hparams


def init_csv():
    ''' create new csv file w header columns for this run '''
    
    cols = ['trial_id', 'file_id', 'ssim_dc', 'psnr_dc', 'ssim_est', \
            'psnr_est', 'alpha_fm', 'num_iter', \
            'iter_start_fm_loss', 'weight_method', 'downsamp_method']

    ct = str(datetime.datetime.now(tz=pytz.timezone('US/Pacific')))
    timestamp = ct.split('.')[0].replace('-', '')\
                                .replace(':', '').replace(' ', '-')
   
    csv_fn = '/home/vanveen/ConvDecoder/expmt/results/run_{}.csv'.format(timestamp)
    f = open(csv_fn, 'a')
    f.write(','.join(cols) + '\n')
    f.close()

    return csv_fn


if __name__ == '__main__':

    hparams = init_parser()

    for hparam_trial in hparams.trials(NUM_TRIALS):
        train(hparam_trial)
