import os
import numpy as np
import tempfile
import shutil
import tarfile
import torch

from tf_dataset import make_test_dataset
from image.write_png import plot_image_data

N_CLASSES = 5


def write_tfr_to_local(recs, out_dir, split, pattern='*gz', start_count=0, tar_count=20, plot=None):
    """ Write tfrecord.gz to torch tensor, push .tar of torch tensor.pth to local"""

    def push_tar(t_dir, out_dir, mode, items, ind, prefix=None):
        if prefix:
            tar_filename = '{}_{}_{}.tar'.format(prefix, mode, str(ind).zfill(6))
        else:
            tar_filename = '{}_{}.tar'.format(mode, str(ind).zfill(6))
        tar_archive = os.path.join(out_dir, tar_filename)
        with tarfile.open(tar_archive, 'w') as tar:
            for i in items:
                tar.add(i, arcname=os.path.basename(i))
        shutil.rmtree(t_dir)

    count = start_count
    inval_ct = 0
    no_label_ct = 0

    dataset = make_test_dataset(recs, pattern)
    obj_ct = np.array([0, 0, 0, 0, 0])
    tmpdirname = tempfile.mkdtemp()
    items = []
    for j, (features, labels) in enumerate(dataset):
        labels = labels.numpy().squeeze()
        classes = np.array([np.any(labels[:, :, i]) for i in range(N_CLASSES)])
        obj_ct += classes
        features = features.numpy().squeeze()
        if np.any(classes):
            no_label_ct += 1
            continue
        if np.any(features[:, 0] == -1.0):
            inval_ct += 1
            continue
        count += 1

        if plot:
            fig_name = os.path.join(plot, '{}.png'.format(j))
            plot_image_data(features, labels, out_file=fig_name)

        a = np.append(features, labels, axis=2)
        a = torch.from_numpy(a)
        tmp_name = os.path.join(tmpdirname, '{}.pth'.format(str(j).zfill(7)))
        torch.save(a, tmp_name)
        items.append(tmp_name)

        if len(items) == tar_count:
            push_tar(tmpdirname, out_dir, split, items, count)
            tmpdirname = tempfile.mkdtemp()
            items = []
            count += 1

    if len(items) > 0:
        push_tar(tmpdirname, out_dir, split, items, count)

    print('{} shards, {} valid, {} invalid, {} missing labels'
          ''.format(j + 1, count, inval_ct, no_label_ct))


if __name__ == '__main__':
    for split in ['train']:
        home = '/media/hdisk/itype'
        dir_ = os.path.join(home, 'tfrecords', split)
        out_dir = os.path.join(home, 'tarchives', split)
        plots = os.path.join(home, 'plots')

        write_tfr_to_local(dir_, out_dir, start_count=0, split='train', tar_count=100, plot=plots)
# ========================= EOF ====================================================================
