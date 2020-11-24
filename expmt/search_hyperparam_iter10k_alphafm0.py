from test_tube import Experiment
from test_tube import HyperOptArgumentParser
import datetime
import pytz
import json

from train_larger_set import train

NUM_ITER = 10000
GPU_ID = 3 
NUM_TRIALS = 1

def init_parser():

    parser = HyperOptArgumentParser(strategy="random_search")

    parser.opt_list('--alpha_fm', type=float, default=0,
                    options=None, tunable=False,         
                    help='weight on feat_map loss')

    parser.opt_list('--iter_start_fm_loss', type=int, default=0,
                    options=None, tunable=False,
                    help='iteration at which to incorporate fm loss')

    parser.opt_list('--weight_method', type=str, default='all',
                    options=None, tunable=False,
                    help='fm loss on early, later, or all net layers')

    parser.opt_list('--downsamp_method', type=str, default='bicubic',
                    options=None, tunable=False,
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

# fix hparams according to loaded json
if __name__ == '__main__':

    hparams = init_parser()

    for _ in range(4):

        for hparam_trial in hparams.trials(NUM_TRIALS):
           
            hparam_trial.alpha_fm = 0 
            hparam_trial.trial_id = '0000_10k'

            train(hparam_trial)
