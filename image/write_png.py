import numpy as np
from matplotlib import colors

from image import feature_spec

MODE = 'itype'
N_CLASSES = 5
FEATURES_DICT = feature_spec.features_dict()
FEATURES = feature_spec.features()

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['grey', 'blue', 'purple', 'pink', 'green'])


def plot_image_data(x, label=None, out_file=None):
    cmap_label = colors.ListedColormap(['white', 'red', 'green', 'blue', 'purple', 'yellow'])
    cmap_ndvi = colors.ListedColormap(['red', 'orange', 'white', 'green'])

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 10))

    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    rgb = np.dstack([r, g, b]).astype(int)

    ndvi = x[:, :, 5]

    mask = label.sum(0) == 0
    label = label.argmax(-1) + 1
    # label[mask] = 0

    ax[0].imshow(rgb)
    ax[0].set(xlabel='image')

    ax[1].imshow(ndvi)
    ax[1].set(xlabel='image')

    ax[2].imshow(label, cmap=cmap_label)
    ax[2].set(xlabel='label')

    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
