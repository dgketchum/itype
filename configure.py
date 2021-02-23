import os
import json
from pathlib import Path

import torch

from image.statistics import STATS

path = Path(__file__).parents

BANDS = 6
N_CLASSES = 5


def get_config(model='clstm', mode='six_channel'):
    data = '/media/hdisk/itype/pth'

    device_ct = torch.cuda.device_count()

    config = {'model': model,
              'mode': mode,
              'image_size': (256, 256),
              'rdm_seed': 1,
              'display_step': 1000,
              'epochs': 300,
              'num_classes': N_CLASSES,
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
        config['batch_size'] = 4 * device_ct
        config['hidden_size'] = 256
        config['input_dim'] = BANDS
        config['seed'] = 121
        config['lr'] = 0.0025
        config['res_dir'] = os.path.join(path[0], 'models', 'nnet', 'results')
        with open(os.path.join(path[0], 'models', 'unet', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    return config


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
