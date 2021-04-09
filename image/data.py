import os
from glob import glob
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

N_CLASSES = 6
N_BANDS = 6
""" ['R', 'G', 'B', 'N', 'std_ndvi', 'mx_ndvi', 'itype'] """


class ITypeDataset(Dataset):

    def __init__(self, data_dir, mode, transforms=None):
        self.data_dir = data_dir
        self.mode = mode
        self.img_paths = glob(os.path.join(data_dir, '*.pth'))

        # add test to train for two-way train/valid split
        if 'train' in data_dir:
            add_paths = glob(os.path.join(data_dir.replace('train', 'test'), '*.pth'))
            self.img_paths += add_paths

        self.transforms = transforms

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img = torch.load(img_path)
        img = img.permute(2, 0, 1)

        if self.mode == 'grey':
            features = img[0, :, :] * 0.2989 + img[1, :, :] * 0.5870 + img[2, :, :] * 0.1140
            features = features.unsqueeze(0).float()
        elif self.mode == 'rgb':
            features = img[:3, :, :].float()
        elif self.mode == 'rgbn':
            features = img[:4, :, :].float()
        elif self.mode == 'rgbn_snt':
            features = img[:8, :, :].float()
        elif self.mode == 'grey_snt':
            grey = img[0, :, :] * 0.2989 + img[1, :, :] * 0.5870 + img[2, :, :] * 0.1140
            grey = grey.unsqueeze(0).float()
            features = torch.cat([grey, img[4:6, :, :]], dim=0)
        else:
            raise KeyError(
                'Must choose from {} image modes'.format(['grey',
                                                          'rgb',
                                                          'rgbn',
                                                          'rgbn_snt',
                                                          'grey_snt']))

        label = img[-1, :, :].long()

        if not torch.isfinite(img).all():
            print('non-finite in {}'.format(img_path))
            if not torch.isfinite(features).all():
                print('lalbel has nan/inf')
            if not torch.isfinite(label).all():
                print('lalbel has nan/inf')

        if self.transforms:
            features = self.transforms(features)
        return features, label

    def __len__(self):
        return len(self.img_paths)


class ITypeDataModule(pl.LightningDataModule):

    def __init__(self, config):
        pl.LightningDataModule.__init__(self, config)
        self.num_workers = config['num_workers']
        self.data_dir = os.path.join(config['dataset_folder'])
        self.batch_sz = config['batch_size']
        self.mode = config['mode']

    def train_dataloader(self):
        train_dir = os.path.join(self.data_dir, 'train')
        train_ds = ITypeDataset(train_dir, self.mode, transforms=None)
        dl = DataLoader(
            train_ds,
            shuffle=True,
            batch_size=self.batch_sz,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=self.collate_fn)
        return dl

    def val_loader(self):
        val_dir = os.path.join(self.data_dir, 'valid')
        valid_ds = ITypeDataset(val_dir, self.mode, transforms=None)
        dl = DataLoader(
            valid_ds,
            shuffle=False,
            batch_size=self.batch_sz,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn)
        return dl

    def test_loader(self):
        test_dir = os.path.join(self.data_dir, 'test')
        test_ds = ITypeDataset(test_dir, self.mode, transforms=None)
        dl = DataLoader(
            test_ds,
            shuffle=False,
            batch_size=self.batch_sz,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn)
        return dl

    @staticmethod
    def collate_fn(data):
        x, y = [], []
        for d in data:
            if d:
                x.append(d[0]), y.append(d[1])
        return torch.stack(x), torch.stack(y)

    def setup(self, stage: Optional[str] = None):
        pass

    def prepare_data(self, *args, **kwargs):
        pass


def get_loader(config, split='train'):
    num_workers = 8
    split_dir = os.path.join(config['dataset_folder'], split)
    batch_sz = config['batch_size']
    mode = config['mode']

    def collate_fn(data):
        x, y = [], []
        for d in data:
            if d:
                x.append(d[0]), y.append(d[1])
        return torch.stack(x), torch.stack(y)

    if split == 'train':
        train_ds = ITypeDataset(split_dir, mode, transforms=None)
        dl = DataLoader(
            train_ds,
            shuffle=True,
            batch_size=batch_sz,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn)
    else:
        valid_ds = ITypeDataset(split_dir, mode, transforms=None)
        dl = DataLoader(
            valid_ds,
            shuffle=False,
            batch_size=batch_sz,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn)

    return dl


def get_loaders(config):
    splits = ['train', 'test', 'valid']
    train, test, valid = (get_loader(config, split) for split in splits)
    return train, test, valid


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
