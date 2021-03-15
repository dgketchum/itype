import os
from copy import deepcopy
from pathlib import Path
from image.data import get_loaders
from predict import plot_prediction
from configure import get_config

path = Path(__file__).parents


def inv_norm(x, config):
    """ get original image data (1 X H x W x T x C ) ==> ( H x W x C )"""
    mean_std = config['norm']
    x = x.squeeze()
    x = x.permute(0, 2, 3, 1)
    # mean, std = torch.tensor(mean_std[0]), torch.tensor(mean_std[1])
    # x = x.mul_(std).add_(mean)
    x = x.detach().numpy()

    return x


def write_test_image(config, plot=False):
    val_loader = get_loaders(config)[2]

    for i, (x, y) in enumerate(val_loader):
        image = inv_norm(deepcopy(x), config)
        y = y.numpy()

        if plot:
            out_fig = os.path.join(config['res_dir'], 'figures', '{}.png'.format(i))
            print('write {}'.format(out_fig))
            plot_prediction(image, y, out_file=out_fig)

        if i > 1:
            break


if __name__ == '__main__':
    root = '/home/dgketchum/itype'
    if not os.path.isdir(root):
        root = '/nobackup/dketchu1/itype'
    data = os.path.join(root, 'pth_snt', '2019')
    plots = os.path.join(root, 'images')
    get_config('unet')
# ========================= EOF ====================================================================
