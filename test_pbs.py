import os
import json
from datetime import datetime
from argparse import ArgumentParser


from models.unet.unet import UNet
from configure import get_config


def prepare_output(config):
    dt = datetime.now().strftime('{}-%Y.%m.%d.%H.%M-{}-{}'.format(config.machine,
                                                                  config.model,
                                                                  config.mode))
    new_dir = os.path.join(config.res_dir, dt)
    os.makedirs(new_dir, exist_ok=True)
    os.makedirs(os.path.join(new_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(new_dir, 'figures'), exist_ok=True)
    with open(os.path.join(new_dir, 'config.json'), 'w') as file:
        file.write(json.dumps(vars(config), indent=4))
    return new_dir


def main(params):
    config = get_config(**vars(params))

    model = UNet(config)
    for k, v in model.hparams.items():
        print(k, v)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model', default='unet')
    parser.add_argument('--mode', default='grey_snt')
    parser.add_argument('--gpu', default='RTX')
    parser.add_argument('--machine', default='pc')
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--progress', default=0, type=int)
    parser.add_argument('--workers', default=12, type=int)
    args = parser.parse_args()
    main(args)
# ========================= EOF ====================================================================
