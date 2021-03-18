import os
from pathlib import Path

import torch

path = Path(__file__).parents

BANDS = 6
N_CLASSES = 6


def get_config(model='unet', mode='six_channel'):
    data = '/media/nvm/itype/pth_snt/2019'
    if not os.path.isdir(data):
        data = '/nobackup/dketchu1/itype/pth_snt/2019'

    device_ct = torch.cuda.device_count()
    print('device count: {}'.format(device_ct))
    node_ct = 0

    config = {'model': model,
              'mode': mode,
              'dataset_folder': data,
              'rdm_seed': 1,
              'epochs': 100,
              'num_classes': N_CLASSES,
              'device_ct': device_ct,
              'device': 'cuda:0',
              'node_ct': node_ct,
              'num_workers': 1,
              'input_dim': BANDS,
              'batch_size': 24 * device_ct,
              'sample_n': [17227321802,
                           1737714929,
                           813083261,
                           1876868565,
                           6397630789,
                           3290628014],
              'res_dir': os.path.join(path[0], 'models', model, 'results'),
              }

    return config


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
