import os
import json
from pathlib import Path

import torch

from image.statistics import STATS

path = Path(__file__).parents

BANDS = 6
N_CLASSES = 6


def get_config(model='unet', mode='six_channel'):
    data = '/home/dgketchum/itype/pth_snt/2019'
    if not os.path.isdir(data):
        data = '/nobackup/dketchu1/itype/pth_snt/2019'

    device_ct = torch.cuda.device_count()

    config = {'model': model,
              'mode': mode,
              'image_size': (256, 256),
              'rdm_seed': 1,
              'display_step': 1000,
              'epochs': 100,
              'num_classes': N_CLASSES,
              'device_count': device_ct,
              'device': 'cuda:0',
              'num_workers': 1,
              'pooling': 'mean_std',
              'dropout': 0.2,
              'gamma': 1,
              'alpha': None,
              'prediction_dir': os.path.join(data, 'test'),
              'norm': STATS[mode], }

    if config['model'] == 'unet':
        config['dataset_folder'] = data

        config['batch_size'] = 12 * device_ct
        config['input_dim'] = BANDS
        config['sample_n'] = [17227321802,
                              1737714929,
                              813083261,
                              1876868565,
                              6397630789,
                              3290628014]
        config['seed'] = 121
        config['lr'] = 0.0001
        config['res_dir'] = os.path.join(path[0], 'models', config['model'], 'results')
        with open(os.path.join(path[0], 'models', config['model'], 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    return config


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
