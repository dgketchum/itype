import os
import numpy as np
from copy import deepcopy

import torch
from matplotlib import pyplot as plt
from matplotlib import colors

from learning.metrics import get_conf_matrix, confusion_matrix_analysis
from models.model_init import get_model
from image.data import get_loaders
from configure import get_config


def pprint_confusion(a):
    np.set_printoptions(precision=0)
    np.seterr(divide='ignore')
    a = np.log10(a)
    a[np.isinf(a)] = 0.0
    print(a)


def inv_norm(x, config):
    """ get original image data (1 X H x W x T x C ) ==> ( H x W x C )"""
    mean_std = config['norm']
    x = x.squeeze()
    x = x.permute(0, 2, 3, 1)
    # mean, std = torch.tensor(mean_std[0]), torch.tensor(mean_std[1])
    # x = x.mul_(std).add_(mean)
    x = x.detach().numpy()
    return x


def predict(config, plot=False):
    device = torch.device(config['device'])

    n_class = config['num_classes']
    confusion = np.zeros((n_class, n_class))

    val_loader = get_loaders(config)[2]
    model = get_model(config)
    check_pt = torch.load(os.path.join(config['res_dir'], 'model.pth.tar'))
    optimizer = torch.optim.Adam(model.parameters())
    model.load_state_dict(check_pt['state_dict'], strict=False)

    model.to(device)
    optimizer.load_state_dict(check_pt['optimizer'])
    model.eval()

    for i, (x, y) in enumerate(val_loader):
        image = inv_norm(deepcopy(x), config)
        x = x.to(device)

        with torch.no_grad():
            out = model(x)
            pred_img = torch.argmax(out, dim=1).cpu().numpy()
            pred_flat = pred_img.flatten()

        y = y.numpy()
        mask = (y > 0).flatten()
        y_flat = y.flatten()

        if plot:
            out_fig = os.path.join(config['res_dir'], 'figures', '{}.png'.format(i))
            print('write {}'.format(out_fig))
            plot_prediction(image, pred_img, y, out_file=out_fig)

        confusion += get_conf_matrix(y_flat[mask], pred_flat[mask], n_class)

    _, overall = confusion_matrix_analysis(confusion)
    f1 = overall['f1-score']
    pprint_confusion(confusion[-5:, -5:])
    print('F1 {:.2f},'.format(f1))


def plot_prediction(x, pred, label, out_file=None):
    cmap_label = colors.ListedColormap(['white', 'green', 'yellow', 'blue', 'pink', 'grey'])
    bounds_l = [0, 1, 2, 3, 4, 5]
    bound_norm_l = colors.BoundaryNorm(bounds_l, len(bounds_l))

    cmap_pred = colors.ListedColormap(['green', 'yellow', 'blue', 'pink', 'grey'])
    bounds_p = [1, 2, 3, 4, 5]
    bound_norm_p = colors.BoundaryNorm(bounds_p, len(bounds_p))

    batch_sz = x.shape[0]
    for i in range(batch_sz):
        a = x[i, :, :]
        fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20, 10))
        r, g, b = a[:, :, 0].astype('uint8'), a[:, :, 1].astype('uint8'), a[:, :, 2].astype('uint8')
        rgb = np.dstack([r, g, b])

        mx_ndvi = a[:, :, 5]
        std_ndvi = a[:, :, 4]

        label_ = label[i, :, :]
        pred_ = pred[i, :, :]

        ax[0].imshow(rgb)
        ax[0].set(xlabel='image')

        ax[1].imshow(mx_ndvi, cmap='RdYlGn')
        ax[1].set(xlabel='max_ndvi')

        ax[2].imshow(std_ndvi, cmap='cool')
        ax[2].set(xlabel='std_ndvi')

        ax[3].imshow(label_, cmap=cmap_label, norm=bound_norm_l)
        ax[3].set(xlabel='label {}'.format(np.unique(label_)))

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
    predict(config, plot=True)
# ========================= EOF ====================================================================
