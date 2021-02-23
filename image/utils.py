import os
import pickle as pkl

import numpy as np
import rasterio
from matplotlib import colors
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from image.data import one_hot_label, N_CLASSES
from image.data import ITypeDataset, build_databunch

MODE = 'itype'
N_CLASSES = 5

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def read_tif(in_, plot=None):
    """ Read geotiff to image and label ndarray"""

    count = 0
    inval_ct = 0
    no_label_ct = 0
    nan_ct = 0

    obj_ct = np.array([0, 0, 0, 0, 0])
    files_ = [os.path.join(in_, x) for x in os.listdir(in_) if x.endswith('.tif')]
    for j, f in enumerate(files_):
        with rasterio.open(f, 'r') as src:
            img = src.read()

        img = img.transpose(1, 2, 0)
        label, features = img[:, :, -1], img[:, :, :-1]
        label = one_hot_label(label, N_CLASSES)
        classes = np.array([np.any(label[:, :, i]) for i in range(N_CLASSES)])
        obj_ct += classes

        if np.isnan(np.sum(features)):
            nan_ct += 1
            print(f)
            continue
        if not np.any(classes):
            no_label_ct += 1
        if np.any(features[:, 0] == -1.0):
            inval_ct += 1

        count += 1

        fig_name = os.path.join(plot, '{}.png'.format(j))
        plot_image_data(features, label, out_file=fig_name)

    print('{} shards, {} valid, {} invalid, {} missing labels'
          ''.format(j + 1, count, inval_ct, no_label_ct))


def plot_image_data(x, label=None, out_file=None):
    cmap_label = colors.ListedColormap(['white', 'green', 'yellow', 'blue', 'pink', 'grey'])
    bounds = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, len(bounds))

    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(20, 10))

    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    rgb = np.dstack([r, g, b]).astype(int)

    mask = label.sum(-1) == 0
    label = label.argmax(-1) + 1
    label[mask] = 0

    mx_ndvi = x[:, :, 5]
    std_ndvi = x[:, :, 4]

    ax[0].imshow(rgb)
    ax[0].set(xlabel='rgb image')

    ax[1].imshow(mx_ndvi, cmap='RdYlGn')
    ax[1].set(xlabel='mx_ndvi')

    ax[2].imshow(std_ndvi, cmap='cool')
    ax[2].set(xlabel='std_ndvi')

    ax[3].imshow(label, cmap=cmap_label, norm=norm)
    ax[3].set(xlabel='label')

    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
        plt.close()
    else:
        plt.show()


def get_transforms(in_, out_norm):
    transform = Compose([ToTensor()])
    valid_ds = ITypeDataset(in_, transforms=transform)
    dl = DataLoader(
        valid_ds,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        collate_fn=None)

    nimages = 0
    mean = 0.
    std = 0.
    for x, _ in dl:
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        nimages += x.size(0)
        mean += x.mean(2).sum(0)
        std += x.std(2).sum(0)

    mean /= nimages
    std /= nimages

    pkl_name = os.path.join(out_norm, 'meanstd.pkl')
    print((mean, std))
    with open(pkl_name, 'wb') as handle:
        pkl.dump((mean, std), handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    home = '/media/hdisk/itype'
    # for split in ['train']:
    #     dir_ = os.path.join(home, 'tif', split)
    #     plots = os.path.join(home, 'plots', split)
    #     read_tif(dir_, plots)

    norms = os.path.join(home, 'normalize')
    dir_ = os.path.join(home, 'tif', 'train')
    get_transforms(dir_, norms)
# ========================= EOF ====================================================================
