import os
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import Trainer

from models.unet.unet import UNet
from configure import get_config


def main(params):
    config = get_config(**vars(params))

    checkpoint_dir = os.path.join(params.checkpoint, 'checkpoints')
    figures_dir = os.path.join(params.checkpoint, 'figures')
    checkpoint = [os.path.join(checkpoint_dir, x) for x in os.listdir(checkpoint_dir)][0]

    model = UNet.load_from_checkpoint(checkpoint_path=checkpoint)
    model.freeze()
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
        out = model(x)  # .permute(0, 2, 3, 1)
        pred = out.argmax(1)
        x, y, pred = x.squeeze().numpy(), y.squeeze().numpy(), pred.squeeze().numpy()
        fig = os.path.join(figures_dir, '{}.png'.format(i))
        plot_prediction(x, y, pred, model.mode, out_file=None)


def plot_prediction(x, label, pred, mode, out_file=None):
    cmap_label = colors.ListedColormap(['white', 'green', 'yellow', 'blue', 'pink', 'grey'])
    bounds_l = [0, 1, 2, 3, 4, 5, 6]
    bound_norm_l = colors.BoundaryNorm(bounds_l, len(bounds_l))

    classes = ['flood', 'sprinkler', 'pivot', 'rainfed', 'uncultivated']
    cmap_pred = colors.ListedColormap(['green', 'yellow', 'blue', 'pink', 'grey'])

    fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20, 10))

    r, g, b = x[0, :, :].astype('uint8'), x[1, :, :].astype('uint8'), x[2, :, :].astype('uint8')
    rgb = np.dstack([r, g, b])
    im = ax[0].imshow(rgb)
    ax[0].set(xlabel='image')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('bottom', size='10%', pad=0.6)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')

    mx_ndvi = x[5, :, :] / 1000.
    im = ax[1].imshow(mx_ndvi, cmap='RdYlGn')
    ax[1].set(xlabel='max_ndvi')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('bottom', size='10%', pad=0.6)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')

    std_ndvi = x[4, :, :] / 1000.
    im = ax[2].imshow(std_ndvi, cmap='cool')
    ax[2].set(xlabel='std_ndvi')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('bottom', size='10%', pad=0.6)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')

    im = ax[3].imshow(label, cmap=cmap_label, norm=bound_norm_l)
    ax[3].set(xlabel='label {}'.format(np.unique(label)))
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes('bottom', size='10%', pad=0.6)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')

    im = ax[4].imshow(pred, cmap=cmap_pred, norm=None)
    ax[4].set(xlabel='pred {}'.format(np.unique(pred)))
    divider = make_axes_locatable(ax[4])
    cax = divider.append_axes('bottom', size='10%', pad=0.6)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.set_xticklabels(classes)


    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
        plt.close()
    else:
        plt.show()
        exit()


if __name__ == '__main__':
    project = '/home/dgketchum/PycharmProjects/itype'
    checkpoint_pth = os.path.join(project, 'models/unet/nas/cas-2021.03.24.11.02-unet-rgbn_snt')
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model', default='unet')
    parser.add_argument('--mode', default='rgbn')
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
