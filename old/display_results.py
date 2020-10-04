''' load results in stored npy files for 20200925 expmt re variance across runs 
    
    data stored in flat length-64 runs via loops 
    [NUM_RUNS=8]*[len(file_id_list)=4]*[len(NUM_ITER_LIST)=2] '''

import numpy as np

NUM_RUNS = 8

list_oi = psnr_list # list of interest
list_name_list = ['psnr', 'ssim']
ii = 1

for idx, list_oi in enumerate([psnr_list, ssim_list]):
    print('')
    print(list_name_list[idx])

    for ii in range(len(NUM_ITER_LIST)):

        print('')
        print('NUM_ITER {}'.format(NUM_ITER_LIST[ii]))

        for ff in range(len(set(file_id_list))):

            list_compare = []

            for rr in range(NUM_RUNS):

                val = list_oi[8*rr + 2*ff + ii]
                list_compare.append(val)

#                 if ii == 0 and ff == 3:
#                     print(val)

            list_compare = np.asarray(list_compare)
            print('id {}: ~N({}, {})'.format(file_id_list[ff], \
                  np.round(list_compare.mean(),4), np.round(list_compare.std(),4)))
