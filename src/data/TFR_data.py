# -*- encoding: utf-8 -*-
"""Layout dataset
"""
import os
import torch
import scipy.io as sio
import numpy as np
from torchvision.datasets import VisionDataset

import pytorch_lightning as pl
import pdb

TOL = 1e-14

class LoadResponse(VisionDataset):
    """Some Information about LoadResponse dataset"""

    def __init__(
        self,
        root,
        loader,
        list_path,
        load_name="u_obs",
        resp_name="u",
        layout_name="F",
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.list_path = list_path
        self.loader = loader
        self.load_name = load_name
        self.resp_name = resp_name
        self.layout_name = layout_name
        self.extensions = extensions
        self.sample_files = make_dataset_list(
            root, list_path, extensions, is_valid_file
        )

    def __getitem__(self, index):
        path = self.sample_files[index]
        load, resp, _ = self.loader(path, self.load_name, self.resp_name)

        load[np.where(load < TOL)] = 298

        if self.transform is not None:
            load = self.transform(load)
        if self.target_transform is not None:
            resp = self.target_transform(resp)

        return load, resp

    def _layout(self):
        path = self.sample_files[0]
        _, _, layout = self.loader(
            path, self.load_name, self.resp_name, self.layout_name
        )
        return layout

    def __len__(self):
        return len(self.sample_files)



class LoadVecResponse(VisionDataset):
    def __init__(
        self,
        root,
        loader,
        list_path,
        load_name="u_obs",
        resp_name="u",
        layout_name="F",
        div_num=4,
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
    ):
        super().__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.loader = loader
        self.list_path = list_path
        self.load_name = load_name
        self.resp_name = resp_name
        self.layout_name = layout_name
        self.div_num = div_num
        self.sample_files = make_dataset_list(
            root, list_path, extensions, is_valid_file
        )

    def __getitem__(self, index):

        path = self.sample_files[index]
        x_context, y_context, x_target, y_target, resp = self._loader(path)

        if self.transform is not None:
            y_context = (y_context - self.transform[0]) / self.transform[1]
        else:
            pass

        if self.target_transform is not None:
            y_target = (y_target - self.target_transform[0]) / self.transform[1]
            resp = (resp - self.target_transform[0]) / self.transform[1]
        else:
            pass

        return (
            x_context,
            y_context.type(torch.FloatTensor),
            x_target,
            y_target.type(torch.FloatTensor),
            resp.type(torch.FloatTensor),
        )

    def __len__(self):
        return len(self.sample_files)

    def _loader(self, path):

        load, resp, _ = self.loader(path, self.load_name, self.resp_name)

        monitor_x, monitor_y = np.where(load > TOL)
        y_context = torch.from_numpy(load[monitor_x, monitor_y].reshape(1, -1)).float()

        monitor_x, monitor_y = monitor_x / load.shape[0], monitor_y / load.shape[1]
        x_context = torch.from_numpy(
            np.concatenate([monitor_x.reshape(-1, 1), monitor_y.reshape(-1, 1)], axis=1)
        ).float()

        x = np.linspace(0, load.shape[0] - 1, load.shape[0]).astype(int)
        y = np.linspace(1, load.shape[1] - 1, load.shape[1]).astype(int)

        x_target = None
        y_target = None
        for i in range(self.div_num):
            for j in range(self.div_num):
                x1, y1 = (
                    x[0 + i : np.size(x) : self.div_num],
                    y[0 + j : np.size(y) : self.div_num],
                )
                x1, y1 = np.meshgrid(x1, y1)
                x_target0 = (
                    torch.from_numpy(
                        np.concatenate([x1.reshape(-1, 1), y1.reshape(-1, 1)], axis=1)
                        / np.max(load.shape)
                    )
                    .float()
                    .unsqueeze(0)
                )
                y_target0 = torch.from_numpy(resp[x1, y1].reshape(1, -1)).unsqueeze(0)
                if x_target is not None:
                    x_target = torch.cat((x_target, x_target0), 0)
                else:
                    x_target = x_target0

                if y_target is not None:
                    y_target = torch.cat((y_target, y_target0), 0)
                else:
                    y_target = y_target0

        y_context = y_context.transpose(0,1)
        x_target = x_target.squeeze(0)
        y_target = y_target.squeeze(0).transpose(0,1)

        return x_context, y_context, x_target, y_target, torch.from_numpy(resp).float()

    def _layout(self):
        path = self.sample_files[0]
        _, _, layout = self.loader(
            path, self.load_name, self.resp_name, self.layout_name
        )
        return layout

    def _inputdim(self):
        path = self.sample_files[0]
        load, _, _ = self.loader(path, self.load_name, self.resp_name, self.layout_name)
        monitor_x, _ = np.where(load > TOL)
        return np.size(monitor_x)


def make_dataset(root_dir, extensions=None, is_valid_file=None):
    """make_dataset() from torchvision."""
    files = []
    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file \
                cannot be None or not None at the same time"
        )
    if extensions is not None:
        is_valid_file = lambda x: has_allowed_extension(x, extensions)

    assert os.path.isdir(root_dir), root_dir
    for root, _, fns in sorted(os.walk(root_dir, followlinks=True)):
        for fn in sorted(fns):
            path = os.path.join(root, fn)
            if is_valid_file(path):
                files.append(path)
    return files


def make_dataset_list(root_dir, list_path, extensions=None, is_valid_file=None):
    """make_dataset() from torchvision."""
    files = []
    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file \
                cannot be None or not None at the same time"
        )
    if extensions is not None:
        is_valid_file = lambda x: has_allowed_extension(x, extensions)

    assert os.path.isdir(root_dir), root_dir
    with open(list_path, "r") as rf:
        for line in rf.readlines():
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            if is_valid_file(path):
                files.append(path)
    return files


def has_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def mat_loader(path, load_name, resp_name=None, layout_name=None):
    mats = sio.loadmat(path)
    load = mats.get(load_name)
    resp = mats.get(resp_name) if resp_name is not None else None
    layout = mats.get(layout_name) if layout_name is not None else None
    return load, resp, layout


class LayoutDataset(LoadResponse):
    """Layout dataset (mutiple files) generated by 'layout-generator'."""

    def __init__(
        self,
        root,
        list_path=None,
        train=True,
        transform=None,
        target_transform=None,
        load_name="u_obs",
        resp_name="u",
    ):
        test_name = os.path.splitext(os.path.basename(list_path))[0]
        subdir = (
            os.path.join("train", "train") if train else os.path.join("test", test_name)
        )

        # find the path of the list of train/test samples
        list_path = os.path.join(root, list_path)

        # find the root path of the samples
        root = os.path.join(root, subdir)

        super().__init__(
            root,
            mat_loader,
            list_path,
            load_name=load_name,
            resp_name=resp_name,
            extensions="mat",
            transform=transform,
            target_transform=target_transform,
        )


class LayoutVecDataset(LoadVecResponse):
    """Layout dataset (mutiple files) generated by 'layout-generator'."""

    def __init__(
        self,
        root,
        list_path=None,
        train=True,
        transform=None,
        div_num=4,
        target_transform=None,
        load_name="u_obs",
        resp_name="u",
    ):
        test_name = os.path.splitext(os.path.basename(list_path))[0]
        subdir = (
            os.path.join("train", "train") if train else os.path.join("test", test_name)
        )

        # find the path of the list of train/test samples
        list_path = os.path.join(root, list_path)

        # find the root path of the samples
        root = os.path.join(root, subdir)

        super().__init__(
            root,
            mat_loader,
            list_path,
            load_name=load_name,
            resp_name=resp_name,
            extensions="mat",
            div_num=div_num,
            transform=transform,
            target_transform=target_transform,
        )



class TFRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        train_path,
        test_path,
        batch_size,
        num_workers,
        mean_heat=298,
        std_heat=50
    ):
        super().__init__()
        self.data_root = data_root
        self.train_path = train_path
        self.test_path = test_path

        self.mean_heat = mean_heat
        self.std_heat = std_heat

        self.batch_size = batch_size
        self.num_workers = num_workers

    def read_vec_data(self):
        trainval_dataset = LayoutVecDataset(
            self.data_root,
            list_path=self.train_path,
            train=True,
            div_num=1,
            transform=[self.mean_heat, self.std_heat],
            target_transform=[self.mean_heat, self.std_heat],
        )
        test_dataset = LayoutVecDataset(
            self.data_root,
            list_path=self.test_path,
            train=False,
            div_num=1,
            transform=[self.mean_heat, self.std_heat],
            target_transform=[self.mean_heat, self.std_heat],
        )
        self.data_info = test_dataset
        return trainval_dataset, test_dataset

    def prepare_data(self):
        self.trainval_dataset_loaded, self.test_dataset_loaded = self.read_vec_data()
        self.default_layout = self.trainval_dataset_loaded._layout()

        self.default_layout = (
            torch.from_numpy(self.default_layout).unsqueeze(0).unsqueeze(0)
        )

    def setup(self, stage=None):

        self.prepare_data()

        # split train/val set
        if stage == 'fit' or stage is None:
            train_length, val_length = int(len(self.trainval_dataset_loaded) * 0.8), len(
                self.trainval_dataset_loaded
            ) - int(len(self.trainval_dataset_loaded) * 0.8)
            train_dataset, val_dataset = torch.utils.data.random_split(
                self.trainval_dataset_loaded, [train_length, val_length]
            )
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            print(
                f"Prepared dataset, train:{int(len(self.train_dataset))},\
                    val:{int(len(self.val_dataset))}"
            )

        if stage == 'fit' or stage is None:
            self.test_dataset = self.test_dataset_loaded
            print(
                f"Prepared dataset, test:{len(self.test_dataset)}"
            )

    def __dataloader(self, dataset, shuffle=False):
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return loader

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.__dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, shuffle=False)



def test_data(dataset_path, test_path, batch_size, return_loader=True):

    test_dataset = LayoutVecDataset(
        dataset_path,
        list_path=test_path,
        train=False,
        div_num=1,
        transform=[298, 50],
        target_transform=[298, 50]
    )

    loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=1,
    )

    return loader
