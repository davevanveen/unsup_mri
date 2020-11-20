from test_tube import Experiment
from test_tube import HyperOptArgumentParser

from train import train

NUM_TRIALS = 5
exp_list = [-2, -3, -4, -5, -10]
ALPHA_FM_LIST = [10**e for e in exp_list]

def init_parser():

    parser = HyperOptArgumentParser(strategy="random_search")

    parser.opt_list('--alpha_fm', type=float, default=0, 
                    options=ALPHA_FM_LIST, tunable=True, 
                    help='weight on feat_map loss')

    hparams = parser.parse_args()

    return hparams

if __name__ == '__main__':

    hparams = init_parser()

    for hparam_trial in hparams.trials(NUM_TRIALS):
        train(hparam_trial)
