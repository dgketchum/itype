import os
from pathlib import Path

import torch

path = Path(__file__).parents

BANDS = 6
N_CLASSES = 6
B_FACTOR = 1


def get_config(model='unet', mode='rgbn_snt', gpu=None):
    if not gpu:
        gpu = 'RTX'

    if gpu == 'V100':
        b_factor = 3
    elif gpu == 'RTX':
        b_factor = 1
    elif gpu == 'K40':
        b_factor = 1.5

    experiments = {'grey': {'input_dim': 1,
                            'batch_size': int(b_factor * 24)},
                   'rgb': {'input_dim': 3,
                           'batch_size': int(b_factor * 48)},
                   'rgbn': {'input_dim': 4,
                            'batch_size': int(b_factor * 36)},
                   'rgbn_snt': {'input_dim': 6,
                                'batch_size': int(b_factor * 24)},
                   'grey_snt': {'input_dim': 3,
                                'batch_size': int(b_factor * 48)}}

    print('{} batch factor: {}'.format(gpu, b_factor))

    data = '/media/nvm/itype/pth_snt/2019'
    if not os.path.isdir(data):
        data = '/nobackup/dketchu1/itype/pth_snt/2019'

    device_ct = torch.cuda.device_count()
    print('device count: {}'.format(device_ct))
    node_ct = 0

    config = {'model': model,
              'mode': mode,
              'input_dim': experiments[mode]['input_dim'],
              'dataset_folder': data,
              'rdm_seed': 1,
              'epochs': 100,
              'lr': 0.0001,
              'n_classes': N_CLASSES,
              'device_ct': device_ct,
              'node_ct': node_ct,
              'num_workers': 1,
              'batch_size': experiments[mode]['batch_size'] * device_ct,
              'sample_n': [0.02032181, 0.20146593, 0.43057132, 0.18652898, 0.054721877, 0.10639013],
              'res_dir': os.path.join(path[0], 'models', model, 'results'),
              }
    print('batch size: {}'.format(config['batch_size']))

    return config


if __name__ == '__main__':
    get_config('unet', 'rgbn_snt')
# ========================= EOF ====================================================================
