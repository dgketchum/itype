import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip

N_CLASSES = 6
N_BANDS = 6
""" ['R', 'G', 'B', 'N', 'std_ndvi', 'mx_ndvi', 'itype'] """


class ITypeDataset(Dataset):

    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.img_paths = glob(os.path.join(data_dir, '*.pth'))
        self.transforms = transforms

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img = torch.load(img_path)
        img = img.permute(2, 0, 1)
        features = img[:6, :, :].float()
        label = img[-1, :, :].long()
        if self.transforms:
            features = self.transforms(features)
        return features, label

    def __len__(self):
        return len(self.img_paths)


def get_loader(config, split='train'):

    num_workers = 8
    split_dir = os.path.join(config['dataset_folder'], split)
    mean, std = config['norm']
    batch_sz = config['batch_size']

    # consider albumentation library
    data_transforms = {
        'train': Compose([
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
            num_workers=num_workers,
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


def get_loaders(config):
    splits = ['train', 'test', 'valid']
    train, test, valid = (get_loader(config, split) for split in splits)
    return train, test, valid


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
