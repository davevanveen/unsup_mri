from test_tube import Experiment
from test_tube import HyperOptArgumentParser
import datetime
import pytz
import json

from train import train

NUM_ITER = 10000
GPU_ID = 3 
NUM_TRIALS = 1

exp_list = [-4]
ALPHA_FM_LIST = [10**e for e in exp_list]

def init_parser():

    parser = HyperOptArgumentParser(strategy="random_search")

    parser.opt_list('--alpha_fm', type=float, default=0, 
                    options=ALPHA_FM_LIST, tunable=True, 
                    help='weight on feat_map loss')

    parser.add_argument('--weight_method', type=str, default='all',
                    help='fm loss on early, later, or all net layers')

    parser.add_argument('--downsamp_method', type=str, default='bilinear',
                    help='interpolation method on ksp_masked')

    parser.add_argument('--num_iter', type=int, default=NUM_ITER,
                        help='number of gradient descent iterations')

    csv_fn = init_csv()
    parser.add_argument('--csv_fn', type=str, default=csv_fn,
                        help='csv filename for results')

    parser.add_argument('--trial_id', type=str, default=None,
                        help='trial_id for unique hparam configs')

    parser.add_argument('--gpu_id', type=int, default=GPU_ID,
                        help='gpu to use for training')

    hparams = parser.parse_args()

    return hparams


def init_csv():
    ''' create new csv file w header columns for this run '''
    
    cols = ['trial_id', 'file_id', 'ssim_dc', 'psnr_dc', 'ssim_est', \
            'psnr_est', 'alpha_fm', 'num_iter', 'weight_method', 'downsamp_method']

    ct = str(datetime.datetime.now(tz=pytz.timezone('US/Pacific')))
    timestamp = ct.split('.')[0].replace('-', '')\
                                .replace(':', '').replace(' ', '-')
   
    csv_fn = '/home/vanveen/ConvDecoder/expmt/results/run_{}.csv'.format(timestamp)
    f = open(csv_fn, 'a')
    f.write(','.join(cols) + '\n')
    f.close()

    return csv_fn

# default way of using this script
if __name__ == '__main__':

    hparams = init_parser()

    for hparam_trial in hparams.trials(NUM_TRIALS):
        hparam_trial.trial_id = 'autybby9'
        train(hparam_trial)
