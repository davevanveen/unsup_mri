''' miscellaneous helper functions '''

import numpy as np

def num_params(net):
    ''' given network, return total number of params '''
    return sum([np.prod(list(p.size())) for p in net.parameters()]);
