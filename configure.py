import os
from pathlib import Path
from argparse import Namespace

import torch

path = Path(__file__).parents

N_CLASSES = 6


def get_config(**params):
    params = Namespace(**params)

    gpu_map = {'V100': 1.5,
               'RTX': 1,
               'K40': 1.5}
    batch = gpu_map[params.gpu]

    experiments = {'grey': {'n_channels': 1,
                            'batch_size': int(batch * 24)},
                   'rgb': {'n_channels': 3,
                           'batch_size': int(batch * 24)},
                   'rgbn': {'n_channels': 4,
                            'batch_size': int(batch * 24)},
                   'rgbn_snt': {'n_channels': 8,
                                'batch_size': int(batch * 24)},
                   'grey_snt': {'n_channels': 3,
                                'batch_size': int(batch * 24)}}

    print('{} batch factor: {}'.format(params.gpu, batch))

    data = '/media/nvm/itype_/pth_snt/2019'
    if not os.path.isdir(data):
        data = '/nobackup/dketchu1/itype/pth_snt/2019'
    if not os.path.isdir(data):
        data = '/home/ubuntu/itype/pth_snt/2019'

    device_ct = torch.cuda.device_count()
    print('device count: {}'.format(device_ct))

    config = {'model': params.model,
              'mode': params.mode,
              'n_channels': experiments[params.mode]['n_channels'],
              'dataset_folder': data,
              'rdm_seed': 1,
              'epochs': 100,
              'lr': 0.0013,
              'unet_dim_seed': 32,
              'n_classes': N_CLASSES,
              'device_ct': device_ct,
              'node_ct': params.nodes,
              'num_workers': params.workers,
              'machine': params.machine,
              'batch_size': experiments[params.mode]['batch_size'] * device_ct * params.nodes,
              'sample_n': [0.02032181, 0.20146593, 0.43057132, 0.18652898, 0.054721877, 0.10639013],
              'res_dir': os.path.join(path[0], 'models', params.model, 'results'),
              }

    print('batch size: {}'.format(config['batch_size']))

    return Namespace(**config)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
