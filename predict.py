import os
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib import colors
from pytorch_lightning import Trainer

from models.unet.unet import UNet
from configure import get_config


def main(params):
    config = get_config(**vars(params))

    path = Path(params.checkpoint)
    checkpoint_dir = os.path.join(config.res_dir, path.parts[-3])

    model = UNet.load_from_checkpoint(checkpoint_path=params.checkpoint)
    model.hparams.dataset_folder = '/media/nvm/itype/pth_snt/2019'
    model.hparams.batch_size = 1

    if params.metrics:
        trainer = Trainer(
            precision=16,
            gpus=config.device_ct,
            num_nodes=config.node_ct,
            log_every_n_steps=5)

        trainer.test(model)

    loader = model.test_dataloader()
    for i, (x, y) in enumerate(loader):
        out = model(x)
        pred = out.argmax(1)
        x, y, pred = x.squeeze().numpy(), y.squeeze().numpy(), pred.squeeze().numpy()
        fig = os.path.join(checkpoint_dir, 'figures', '{}.png'.format(i))
        plot_prediction(x, y, pred, model.mode, out_file=fig)


def plot_prediction(x, label, pred, mode, out_file=None):
    cmap_label = colors.ListedColormap(['white', 'green', 'yellow', 'blue', 'pink', 'grey'])
    bounds_l = [0, 1, 2, 3, 4, 5]
    bound_norm_l = colors.BoundaryNorm(bounds_l, len(bounds_l))

    cmap_pred = colors.ListedColormap(['green', 'yellow', 'blue', 'pink', 'grey'])
    bounds_p = [1, 2, 3, 4, 5]
    bound_norm_p = colors.BoundaryNorm(bounds_p, len(bounds_p))

    fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20, 10))

    if 'rgb' in mode:
        r, g, b = x[0, :, :].astype('uint8'), x[1, :, :].astype('uint8'), x[2, :, :].astype('uint8')
        rgb = np.dstack([r, g, b])
        ax[0].imshow(rgb)
        ax[0].set(xlabel='image')

    elif 'grey' in mode:
        ax[0].imshow(x[0, :, :])
        ax[0].set(xlabel='image')

    if 'snt' in mode:
        std_ndvi = x[1, :, :]
        mx_ndvi = x[2, :, :]

        ax[1].imshow(mx_ndvi, cmap='RdYlGn_r')
        ax[1].set(xlabel='max_ndvi')

        ax[2].imshow(std_ndvi, cmap='cool')
        ax[2].set(xlabel='std_ndvi')

    ax[3].imshow(label, cmap=cmap_label, norm=bound_norm_l)
    ax[3].set(xlabel='label {}'.format(np.unique(label)))

    ax[4].imshow(pred, cmap=cmap_pred, norm=bound_norm_p)
    ax[4].set(xlabel='pred {}'.format(np.unique(pred)))

    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    checkpoint_pth = '/home/dgketchum/PycharmProjects/itype/models/unet/results/' \
                     'pc-2021.03.21.10.02-unet-rgbn_snt/checkpoints/epoch=9-step=999.ckpt'
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model', default='unet')
    parser.add_argument('--mode', default='rgbn_snt')
    parser.add_argument('--gpu', default='RTX')
    parser.add_argument('--machine', default='pc')
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--progress', default=0, type=int)
    parser.add_argument('--workers', default=12, type=int)
    parser.add_argument('--checkpoint', default=checkpoint_pth)
    parser.add_argument('--metrics', default=False, type=bool)
    args = parser.parse_args()
    main(args)
# ========================= EOF ====================================================================
