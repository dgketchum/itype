import os
import pickle as pkl

import numpy as np
from matplotlib import pyplot as plt
import torch
import rasterio
from matplotlib import colors
from torch.utils.data import DataLoader

from image.data import N_CLASSES
from image.data import ITypeDataset

SUBSET_SZ = 256


def write_pth_subsets(in_, _out, start_ct=None):
    def tile(a):
        s = SUBSET_SZ
        t = [a[x:x + s, y:y + s] for x in range(0, a.shape[0], s) for y in range(0, a.shape[1], s)]
        return t

    files_ = [os.path.join(in_, x) for x in os.listdir(in_) if x.endswith('.tif')]
    obj_ct = np.array([0, 0, 0, 0, 0, 0])
    ct, no_label = 0, 0
    if start_ct:
        ct = start_ct
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
                f[:, :, -4:] = f[:, :, -4:] * 1000
                stack = np.concatenate([f, l], axis=2).astype(np.int16)
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
        print(f)
        img = src.read()

    img = img.transpose(1, 2, 0)
    label, features = img[:, :, -1].reshape(img.shape[0], img.shape[1], 1), img[:, :, :-1]

    if not np.isfinite(features).all():
        print('image {} has nan/inf'.format(f))
        raise TypeError

    if not np.isfinite(label).all():
        print('label {} has nan/inf'.format(f))
        raise TypeError

    if not np.less_equal(label, N_CLASSES - 1).all():
        print('{} has label greater than {}'.format(f, N_CLASSES - 1))
        raise TypeError

    if not np.greater_equal(label, 0).all():
        print('{} has label less than {}'.format(f, 0))
        raise TypeError

    return features, label


def write_tif_image_plots(in_, out):
    files_ = [os.path.join(in_, x) for x in os.listdir(in_) if x.endswith('.tif')]
    for j, f in enumerate(files_):
        try:
            features, label = read_tif(f)
            fig_name = os.path.join(out, '{}.png'.format(os.path.basename(f)))
            plot_image_data(features, label, out_file=fig_name)
        except TypeError:
            pass
    return None


def write_pth_image_plots(in_, out):
    files_ = [os.path.join(in_, x) for x in os.listdir(in_) if x.endswith('.pth')]
    for j, f in enumerate(files_):
        print(f)
        img = torch.load(f)
        img = img.numpy()
        features = img[:, :, :6].astype(np.float)
        label = img[:, :, -1].astype(np.int)
        fig_name = os.path.join(out, '{}.png'.format(os.path.basename(f)))
        plot_image_data(features, label, out_file=fig_name)
        if j > 10:
            break
    return None


def plot_image_data(x, label=None, out_file=None):
    cmap_label = colors.ListedColormap(['white', 'green', 'yellow', 'blue', 'pink', 'grey'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, len(bounds))

    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(20, 10))

    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    rgb = np.dstack([r, g, b]).astype(int)

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
    train_ds = ITypeDataset(in_, transforms=None)
    dl = DataLoader(
        train_ds,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        collate_fn=None)

    nimages = 0
    mean = 0.
    std = 0.
    first = True
    for i, (x, _) in enumerate(dl):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        if first:
            max_ = torch.amax(x, dim=(2, 0))
            min_ = torch.amin(x, dim=(2, 0))
            first = False

        amax = torch.amax(x, dim=(2, 0))
        amin = torch.amin(x, dim=(2, 0))
        max_ = torch.max(amax, max_)
        min_ = torch.min(amin, min_)
        nimages += x.size(0)
        mean += x.mean(2).sum(0)
        std += x.std(2).sum(0)

    print('channel-wise min: {}'.format(list(min_.numpy())))
    print('channel-wise max: {}'.format(list(max_.numpy())))

    mean /= nimages
    std /= nimages

    print((mean, std))
    pkl_name = os.path.join(out_norm, 'meanstd.pkl')
    with open(pkl_name, 'wb') as handle:
        pkl.dump((mean, std), handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    home = '/media/nvm/itype'
    if not os.path.isdir(home):
        home = '/home/dgketchum/itype'
    instrument = 'snt'
    yr_ = str(2019)
    split = 'train'
    tif_recs = os.path.join(home, 'tif_{}'.format(instrument), yr_, split)
    pth_recs = os.path.join(home, 'pth_{}'.format(instrument), yr_, split)
    write_pth_subsets(tif_recs, pth_recs, start_ct=0)

    # for split in ['test', 'train', 'valid']:
    #     dir_ = os.path.join(home, 'pth_{}'.format(instrument), '2019', split)
    #     pltt = os.path.join(home, 'plot_pth_{}'.format(instrument))
    #     write_pth_image_plots(dir_, pltt)

    # norms = os.path.join(home, 'normalize')
    # dir_ = os.path.join(home, 'pth_{}'.format(instrument), 'train')
    # get_transforms(dir_, norms)
# ========================= EOF ====================================================================
