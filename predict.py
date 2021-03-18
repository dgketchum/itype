import os
from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import colors

from models.unet.unet import UNet
from image.data import get_loaders
from configure import get_config


def predict(config, model_path=None, plot=False):
    device = torch.device(config['device'])
    val_loader = get_loaders(config)[2]

    model = UNet.load_from_checkpoint(model_path, strict=False)
    model.freeze()
    model.to(device)

    for i, (x, y) in enumerate(val_loader):
        image = deepcopy(x).cpu().numpy()
        x = x.to(device)
        out = model(x)
        pred_img = torch.argmax(out, dim=1).cpu().numpy()
        y = y.numpy()

        if plot:
            out_fig = os.path.join(config['res_dir'], 'figures', '{}.png'.format(i))
            print('write {}'.format(out_fig))
            plot_prediction(image, y, pred_img, out_file=out_fig)


def plot_prediction(x, label, pred=None, out_file=None):
    cmap_label = colors.ListedColormap(['white', 'green', 'yellow', 'blue', 'pink', 'grey'])
    bounds_l = [0, 1, 2, 3, 4, 5]
    bound_norm_l = colors.BoundaryNorm(bounds_l, len(bounds_l))

    if isinstance(pred, np.ndarray):
        cmap_pred = colors.ListedColormap(['green', 'yellow', 'blue', 'pink', 'grey'])
        bounds_p = [1, 2, 3, 4, 5]
        bound_norm_p = colors.BoundaryNorm(bounds_p, len(bounds_p))

    batch_sz = x.shape[0]

    for i in range(batch_sz):

        a = x[i, :, :]

        if isinstance(pred, np.ndarray):
            fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20, 10))
        else:
            fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(20, 10))

        r, g, b = a[:, :, 0].astype('uint8'), a[:, :, 1].astype('uint8'), a[:, :, 2].astype('uint8')
        rgb = np.dstack([r, g, b])

        mx_ndvi = a[:, :, 5]
        std_ndvi = a[:, :, 4]

        ax[0].imshow(rgb)
        ax[0].set(xlabel='image')

        ax[1].imshow(mx_ndvi, cmap='RdYlGn')
        ax[1].set(xlabel='max_ndvi')

        ax[2].imshow(std_ndvi, cmap='cool')
        ax[2].set(xlabel='std_ndvi')

        label_ = label[i, :, :]
        ax[3].imshow(label_, cmap=cmap_label, norm=bound_norm_l)
        ax[3].set(xlabel='label {}'.format(np.unique(label_)))

        if isinstance(pred, np.ndarray):
            pred_ = pred[i, :, :]
            ax[4].imshow(pred_, cmap=cmap_pred, norm=bound_norm_p)
            ax[4].set(xlabel='pred {}'.format(np.unique(pred_)))

        out_ = out_file.replace('.png', '_{}.png'.format(i))
        plt.tight_layout()
        if out_file:
            plt.savefig(out_)
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    config = get_config('unet')
    check_dir = '/home/dgketchum/PycharmProjects/itype/models/from_nas'
    checkpoint_path = os.path.join(check_dir, 'model-18MAR2021.pth.tar')
    predict(config, model_path=checkpoint_path, plot=True)
# ========================= EOF ====================================================================
