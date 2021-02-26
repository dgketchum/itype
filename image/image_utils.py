import os
import pickle as pkl

import numpy as np
from matplotlib import pyplot as plt
import torch
import rasterio
from matplotlib import colors
from torch.utils.data import DataLoader

from image.data import N_CLASSES, N_BANDS
from image.data import ITypeDataset

SUBSET_SZ = 256


def write_pth_subsets(in_, _out):
    def tile(a):
        s = SUBSET_SZ
        t = [a[x:x + s, y:y + s] for x in range(0, a.shape[0], s) for y in range(0, a.shape[1], s)]
        return t

    files_ = [os.path.join(in_, x) for x in os.listdir(in_) if x.endswith('.tif')]
    obj_ct = np.array([0, 0, 0, 0, 0, 0])
    ct, no_label = 0, 0
    for j, file_ in enumerate(files_):
        try:
            features, label = read_tif(file_)
            classes = np.array([np.count_nonzero(label == i) for i in range(N_CLASSES)])
            obj_ct += classes
        except TypeError:
            continue
        sub_f, sub_l = tile(features), tile(label)
        for f, l in zip(sub_f, sub_l):
            if np.any(l):
                f[:, :, -2:] = f[:, :, -2:] * 100
                stack = np.concatenate([f, l], axis=2).astype(np.uint8)
                stack = torch.tensor(stack)
                name_ = os.path.join(_out, '{}.pth'.format(ct))
                torch.save(stack, name_)
                ct += 1
            else:
                no_label += 1
    print('{} subsets from {} tif images\n{} chunks w/o label'.format(ct, j, no_label))
    print('class distribution:\n{}'.format(obj_ct))


def read_tif(f):
    """ Read geotiff to image and label ndarray"""

    with rasterio.open(f, 'r') as src:
        img = src.read()

    img = img.transpose(1, 2, 0)
    label, features = img[:, :, -1].reshape(img.shape[0], img.shape[1], 1), img[:, :, :-1]

    if np.isnan(np.sum(features)):
        print('{} has nan'.format(f))
        raise TypeError

    return features, label


def write_image_plots(in_, out):
    files_ = [os.path.join(in_, x) for x in os.listdir(in_) if x.endswith('.tif')]
    for j, f in enumerate(files_):
        features, label = read_tif(f)
        fig_name = os.path.join(out, '{}_lst_.png'.format(j))
        plot_image_data(features, label, out_file=fig_name)
    return None


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
    """Run through unnormalized training data for global mean and std."""
    valid_ds = ITypeDataset(in_, transforms=None)
    dl = DataLoader(
        valid_ds,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        collate_fn=None)

    nimages = 0
    mean = 0.
    std = 0.
    for i, (x, _) in enumerate(dl):
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
    # for split in ['train', 'test', 'valid']:
    #     dir_ = os.path.join(home, 'tif', split)
    #     pth = os.path.join(home, 'pth', split)
    #     write_pth_subsets(dir_, pth)

    dir_ = os.path.join(home, 'tif_lst', 'train')
    pltt = os.path.join(home, 'plots')
    write_image_plots(dir_, pltt)

    # norms = os.path.join(home, 'normalize')
    # dir_ = os.path.join(home, 'pth', 'train')
    # get_transforms(dir_, norms)
# ========================= EOF ====================================================================
