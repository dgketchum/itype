import os
from copy import deepcopy
from pathlib import Path
from image.data import get_loaders
from predict import plot_prediction
from configure import get_config

path = Path(__file__).parents


def write_test_image(config, plot=False):
    print(config['device_ct'])
    val_loader = get_loaders(config)[2]

    for i, (x, y) in enumerate(val_loader):
        image = deepcopy(x).numpy()
        y = y.numpy()

        if plot:
            out_fig = os.path.join(config['res_dir'], 'test_figs', '{}.png'.format(i))
            print('write {}'.format(out_fig))
            plot_prediction(image, y, out_file=out_fig)

        if i > 1:
            break


if __name__ == '__main__':
    config = get_config(gpu='K40')
    write_test_image(config, plot=True)
# ========================= EOF ====================================================================
