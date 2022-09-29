from functools import partial
import os
import torch
import scipy.io as sio
import numpy as np

import pytorch_lightning as pl
import pdb


class D30Dataset(torch.utils.data.Dataset):
    def __init__(self, num=1024, p_num=256, test=False):
        super(D30Dataset, self).__init__()

        self.dim = 30

        if self.dim == 10:
            self.sigma_range = [1, 2]
        elif self.dim == 20:
            self.sigma_range = [2, 4]
        elif self.dim == 30:
            self.sigma_range = [4, 8]
        else:
            raise NotImplementedError

        self.support = [-1, 1]
        self.kernel_num = 5
        self.p_num = p_num

        self.gaussian_mu = torch.rand((num, self.kernel_num, self.dim)) * (self.support[1]-self.support[0]) + self.support[0]

        sigma = torch.rand((num, self.kernel_num)) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
        self.sigma_2 = torch.pow(sigma, 2)
        self.A = torch.rand((num, self.kernel_num)) * 2 - 1

    def __len__(self):
        return len(self.gaussian_mu)

    def __getitem__(self, index):

        mu = self.gaussian_mu[index][None, ...]
        sigma = self.sigma_2[index][None, ...]
        A = self.A[index][None, ...]

        x = torch.rand((self.p_num, 1, self.dim)) * (self.support[1]-self.support[0]) + self.support[0]
        x_norm_2 = torch.pow(x - mu, 2).sum(2)
        y = (A * torch.exp(-x_norm_2/sigma)).sum(1)
        return torch.cat([x.squeeze(1), y[...,None]], axis=1)


def _collate(data, test=True):
    data = torch.stack(data)
    O, T = data[:, :64, :], data[:, 64:, :]
    return O[...,0:-1], O[...,-1:], T[...,0:-1], T[...,-1:], 64


class D30DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        train_path,
        test_path,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_path = train_path
        self.test_path = test_path

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.train_dataset = D30Dataset(1024 * 256)
        self.test_dataset = D30Dataset(512, test=True)

    def setup(self, stage=None):
        self.prepare_data()

    def __dataloader(self, dataset, shuffle=False, test=False):
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            collate_fn=partial(_collate, test=test),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return loader

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.__dataloader(self.test_dataset, shuffle=False, test=True)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, shuffle=False, test=True)
