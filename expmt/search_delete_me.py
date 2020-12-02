from test_tube import Experiment
from test_tube import HyperOptArgumentParser
import datetime
import pytz
import json

from train_delete_me import train

NUM_ITER = 100
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

    parser.add_argument('--trial_id', type=str, default=None,
                        help='trial_id for unique hparam configs')

    parser.add_argument('--gpu_id', type=int, default=GPU_ID,
                        help='gpu to use for training')

    hparams = parser.parse_args()

    return hparams


def init_csv():
    ''' create new csv file w header columns for this run '''
    
    return None

# default way of using this script
if __name__ == '__main__':

    hparams = init_parser()

    for hparam_trial in hparams.trials(NUM_TRIALS):
        train(hparam_trial)
