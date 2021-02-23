import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip
import rasterio

N_CLASSES = 5
""" ['R', 'G', 'B', 'N', 'std_ndvi', 'mx_ndvi', 'itype'] """


class ITypeDataset(Dataset):

    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.img_paths = glob(os.path.join(data_dir, '*.tif'))
        self.transforms = transforms

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img = rasterio.open(img_path, 'r').read()
        img = img.transpose(1, 2, 0)
        label, features = img[:, :, -1], img[:, :, :-1]
        label = one_hot_label(label, N_CLASSES)
        return features, label

    def __len__(self):
        return len(self.img_paths)


def build_databunch(data_dir, img_sz, batch_sz, norms, mode='train'):
    num_workers = 0

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    mean, std = norms
    data_transforms = {
        'train': Compose([
            RandomResizedCrop(img_sz),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean, std)]),
        'valid': Compose([
            ToTensor(),
            Normalize(mean, std)]),
    }

    if mode == 'train':
        train_ds = ITypeDataset(train_dir, transforms=data_transforms['train'])
        dl = DataLoader(
            train_ds,
            shuffle=False,
            batch_size=batch_sz,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn)
    else:
        valid_ds = ITypeDataset(valid_dir, transforms=data_transforms['valid'])
        dl = DataLoader(
            valid_ds,
            batch_size=batch_sz,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn)

    return dl


def collate_fn(data):
    x, y = [], []
    for d in data:
        if d:
            x.append(d[0]), y.append(d[1])
    return x, y


def one_hot_label(labels, n_classes):
    h, w = labels.shape
    ls = []
    for i in range(n_classes):
        where = np.where(labels != i + 1, np.zeros((h, w)), np.ones((h, w)))
        ls.append(where)
    temp = np.stack(ls, axis=2)
    return temp


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
