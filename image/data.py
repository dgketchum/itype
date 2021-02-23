import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip
import rasterio

N_CLASSES = 5
""" ['R', 'G', 'B', 'N', 'std_ndvi', 'mx_ndvi', 'itype'] """


class ITypeDataset(Dataset):

    def __init__(self, data_dir, transforms):
        self.data_dir = data_dir
        self.img_paths = glob(os.path.join(data_dir, '*.pth'))
        self.transforms = transforms

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img = torch.load(img_path)
        img = img.permute(2, 0, 1)
        features, label = img[:6, :, :].float(), img[6:, :, :]
        label = label.argmax(dim=0)
        label = label.reshape(1, label.shape[0], label.shape[1])
        features = self.transforms(features)
        label = label.argmax(0)
        return features, label

    def __len__(self):
        return len(self.img_paths)


def get_loader(config, split='train'):

    num_workers = 0
    split_dir = os.path.join(config['dataset_folder'], split)
    mean, std = config['norm']
    batch_sz = config['batch_size']

    data_transforms = {
        'train': Compose([
            RandomResizedCrop(config['image_size']),
            RandomHorizontalFlip(),
            Normalize(mean, std)]),
        'valid': Compose([
            Normalize(mean, std)]),
    }

    if split == 'train':
        train_ds = ITypeDataset(split_dir, transforms=data_transforms['train'])
        dl = DataLoader(
            train_ds,
            shuffle=True,
            batch_size=batch_sz,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn)
    else:
        valid_ds = ITypeDataset(split_dir, transforms=data_transforms['valid'])
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
    return torch.stack(x), torch.stack(y)


def one_hot_label(labels, n_classes):
    h, w = labels.shape
    ls = []
    for i in range(n_classes):
        where = np.where(labels != i + 1, np.zeros((h, w)), np.ones((h, w)))
        ls.append(where)
    temp = np.stack(ls, axis=2)
    return temp


def get_loaders(config):
    splits = ['train', 'test', 'valid']
    train, test, valid = (get_loader(config, split) for split in splits)
    return train, test, valid


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
