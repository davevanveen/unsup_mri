''' Various fucntions for data transformations '''

import numpy as np
import torch

def to_tensor(arr):
    ''' convert numpy array to torch tensor
        if arr is complex, real/imag parts stacked on last dimn '''
    if np.iscomplexobj(arr):
        arr = np.stack((arr.real, arr.imag), axis=-1)
    return torch.from_numpy(arr)

def to_np(arr):
    ''' converts torch.Variable to numpy array
        shape [1, C, W, H] --> [C, W, H] '''
    return arr.data.cpu().numpy()[0]

