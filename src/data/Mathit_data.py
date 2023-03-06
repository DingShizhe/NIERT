import os
from re import L

import pytorch_lightning as pl
import pdb

from builtins import NotImplemented, NotImplementedError, input
import numpy as np
import sympy
import warnings
import torch
from torch.utils import data
import math
from .mathit.utils import load_metadata_hdf5, load_eq
from sympy.core.rules import Transform
from sympy import sympify, Float, Symbol
from multiprocessing import Manager

# R_log = lambda x: torch.where(x > 0, torch.log(x + 1.0), -torch.log(-x + 1.0))
R_log = lambda x: x

from numpy import (
    exp,
    # cos,
    # sin,
    nan,
    pi,
    e,
    sqrt,
)

def sin(x):
    if isinstance(x, int) or isinstance(x, float):
        return math.sin(x) if -math.pi * 4 > x > math.pi * 4 else math.nan

    res = torch.sin(x)
    mask = torch.logical_or(x > math.pi * 4, x < -math.pi * 4)
    res[mask] = math.nan
    # if torch.any(mask):
    #     print("dsz_sin fuck!!")
    return res

def cos(x):
    if isinstance(x, int) or isinstance(x, float):
        return math.cos(x) if -math.pi * 4 > x > math.pi * 4 else math.nan

    res = torch.cos(x)
    mask = torch.logical_or(x > math.pi * 4, x < -math.pi * 4)
    res[mask] = math.nan
    # if torch.any(mask):
    #     print("dsz_cos fuck!!")
    return res

import types
from typing import List
import random
from torch.distributions.uniform import Uniform
from .mathit.dataset.data_utils import sample_symbolic_constants
from .mathit.dataset.generator import Generator, UnknownSymPyOperator, ValueErrorExpression
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .mathit.dataset.dclasses import DatasetDetails, Equation
from functools import partial
from ordered_set import OrderedSet
from pathlib import Path
import hydra

import sympy as sp

import sys
# sys.modules['dataset'] = dataset


class NesymresDataset(data.Dataset):
    def __init__(
        self,
        data_path: Path,
        cfg,
        mode: str
    ):  
        #m = Manager()
        #self.eqs = m.dict({i:eq for i, eq in enumerate(data.eqs)})
        metadata = load_metadata_hdf5(hydra.utils.to_absolute_path(data_path))
        cfg.total_variables = metadata.total_variables
        cfg.total_coefficients = metadata.total_coefficients
        self.len = metadata.total_number_of_eqs
        self.eqs_per_hdf = metadata.eqs_per_hdf
        self.word2id = metadata.word2id
        self.data_path = data_path
        self.mode = mode
        self.cfg = cfg


    def __getitem__(self, index):
        eq = load_eq(self.data_path, index, self.eqs_per_hdf)

        code = types.FunctionType(eq.code, globals=globals(), name="f")
        consts, initial_consts = sample_symbolic_constants(eq, self.cfg.constants)

        if self.cfg.predict_c:
            eq_string = eq.expr.format(**consts)
        else:
            eq_string = eq.expr.format(**initial_consts)

        # import time
        # print(eq_string)
        # time.sleep(0.5)

        # pdb.set_trace()

        # eq_sympy_infix, eq_string_dsz = constants_to_placeholder(eq_string)
        # eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)

        # # t = tokenize(eq_sympy_prefix,self.word2id)
        # curr = Equation(code=code,expr=eq_sympy_infix,expr_dsz=eq_string_dsz,coeff_dict=consts,variables=eq.variables,support=eq.support, tokenized=None, valid=True)

        # return curr

        try:
            eq_sympy_infix, eq_string_dsz = constants_to_placeholder(eq_string)
            eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)
        except UnknownSymPyOperator as e:
            print(e)
            return Equation(code=code,expr=[],coeff_dict=consts,variables=eq.variables,support=eq.support, valid=False)
        except RecursionError as e:
            print(e)
            return Equation(code=code,expr=[],coeff_dict=consts,variables=eq.variables,support=eq.support, valid=False)
        except ValueErrorExpression as e:
            print(e)
            return Equation(code=code,expr=[],coeff_dict=consts,variables=eq.variables,support=eq.support, valid=False)

        # try:
        #     # t = tokenize(eq_sympy_prefix,self.word2id)
        #     curr = Equation(code=code,expr=eq_sympy_infix,expr_dsz=eq_string_dsz,coeff_dict=consts,variables=eq.variables,support=eq.support, tokenized=t, valid=True)
        # except:
        #     t = []
        #     curr = Equation(code=code,expr=eq_sympy_infix,coeff_dict=consts,variables=eq.variables,support=eq.support, valid=False)

        curr = Equation(code=code,expr=eq_sympy_infix,expr_dsz=eq_string_dsz,coeff_dict=consts,variables=eq.variables,support=eq.support, tokenized=[], valid=True)

        return curr

    def __len__(self):
        return self.len

def constants_to_placeholder(s,symbol="c"):
    sympy_expr = sympify(s)  # self.infix_to_sympy("(" + s + ")")
    sympy_expr_c = sympy_expr.xreplace(
        Transform(
            lambda x: Symbol(symbol, real=True, nonzero=True),
            lambda x: isinstance(x, Float),
        )
    )
    return sympy_expr_c, sympy_expr


def custom_collate_fn(eqs: List[Equation], cfg, DIM, return_n=False) -> List[torch.tensor]:

    filtered_eqs = [eq for eq in eqs if eq.valid]
    # print("filtered_eqs", len(filtered_eqs))
    res = evaluate_and_wrap(filtered_eqs, cfg, DIM, return_n)
    return res

    # if return_n:
    #     res, _, C, return_eqs, mask_p, min_max = res
    #     return res, return_eqs, C, mask_p, min_max
    # else:
    #     res, _, mask_p = res
    #     return res, mask_p


def number_of_support_points(p, type_of_sampling_points, DIM):
    if type_of_sampling_points == "constant":
        curr_p = p
    elif type_of_sampling_points == "logarithm":

        if DIM == 1:
            p == 25
        else:
            p = 50

        if p == 25:
            curr_p = int(5 ** Uniform(1, math.log10(p)/math.log10(5)).sample())
        elif p == 50:
            curr_p = int(10 ** Uniform(1, math.log10(p)/math.log10(10)).sample())
        else:
            raise NotImplementedError
    else:
        raise NameError
    return curr_p


def sample_support(eq, curr_p, cfg):
    sym = []
    if not eq.support:
        distribution =  torch.distributions.Uniform(cfg.fun_support.min,cfg.fun_support.max) #torch.Uniform.distribution_support(cfg.fun_support[0],cfg.fun_support[1])
    else:
        raise NotImplementedError
    
    for sy in cfg.total_variables:
        if sy in eq.variables:
            curr = distribution.sample([int(curr_p)])
        else:
            curr = torch.zeros(int(curr_p))
        sym.append(curr)
    return torch.stack(sym)


def sample_support_dim1(eq, curr_p, cfg):
    sym = []
    if not eq.support:
        distribution =  torch.distributions.Uniform(cfg.fun_support.min,cfg.fun_support.max) #torch.Uniform.distribution_support(cfg.fun_support[0],cfg.fun_support[1])
    else:
        raise NotImplementedError
    
    assert len(cfg.total_variables) == 1

    for sy in cfg.total_variables:
        if sy in eq.variables:
            curr = distribution.sample([int(curr_p)])
            # pdb.set_trace()
            even_curr = torch.linspace(cfg.fun_support.min,cfg.fun_support.max,256)
            curr = torch.cat([curr, even_curr])
        else:
            curr = torch.zeros(int(curr_p))
        sym.append(curr)
    return torch.stack(sym)


def sample_support_dim2(eq, curr_p, cfg):
    sym = []
    if not eq.support:
        distribution = torch.distributions.Uniform(cfg.fun_support.min,cfg.fun_support.max) #torch.Uniform.distribution_support(cfg.fun_support[0],cfg.fun_support[1])
    else:
        raise NotImplementedError

    assert len(cfg.total_variables) == 2

    for sy in cfg.total_variables:
        if sy in eq.variables:
            curr = distribution.sample([int(curr_p)])
        else:
            curr = torch.zeros(int(curr_p))
        sym.append(curr)
    support = torch.stack(sym)
    _even_curr = torch.linspace(cfg.fun_support.min,cfg.fun_support.max,16)

    _even_curr_x, _even_curr_y = torch.meshgrid(_even_curr, _even_curr)

    even_curr = torch.stack([_even_curr_x.flatten(), _even_curr_y.flatten()])

    support = torch.cat([support, even_curr], axis=1)
    return support

def sample_support_dim3(eq, curr_p, cfg):
    sym = []
    if not eq.support:
        distribution = torch.distributions.Uniform(cfg.fun_support.min,cfg.fun_support.max) #torch.Uniform.distribution_support(cfg.fun_support[0],cfg.fun_support[1])
    else:
        raise NotImplementedError
    
    assert len(cfg.total_variables) == 3

    for sy in cfg.total_variables:
        if sy in eq.variables:
            curr = distribution.sample([int(curr_p)])
        else:
            curr = torch.zeros(int(curr_p))
        sym.append(curr)
    support = torch.stack(sym)

    _even_curr = torch.linspace(cfg.fun_support.min,cfg.fun_support.max,8)
    _even_curr_x, _even_curr_y, _even_curr_z = torch.meshgrid(_even_curr, _even_curr, _even_curr)

    even_curr = torch.stack([_even_curr_x.flatten(), _even_curr_y.flatten(), _even_curr_z.flatten()])

    support = torch.cat([support, even_curr], axis=1)
    return support


def sample_support_dim4(eq, curr_p, cfg):
    sym = []
    if not eq.support:
        distribution = torch.distributions.Uniform(cfg.fun_support.min,cfg.fun_support.max) #torch.Uniform.distribution_support(cfg.fun_support[0],cfg.fun_support[1])
    else:
        raise NotImplementedError
    
    assert len(cfg.total_variables) == 4

    for sy in cfg.total_variables:
        if sy in eq.variables:
            curr = distribution.sample([int(curr_p)])
        else:
            curr = torch.zeros(int(curr_p))
        sym.append(curr)
    support = torch.stack(sym)

    _even_curr = torch.linspace(cfg.fun_support.min,cfg.fun_support.max,5)
    _even_curr_x, _even_curr_y, _even_curr_z, _even_curr_w = torch.meshgrid(_even_curr, _even_curr, _even_curr, _even_curr)

    even_curr = torch.stack([
        _even_curr_x.flatten(),
        _even_curr_y.flatten(),
        _even_curr_z.flatten(),
        _even_curr_w.flatten()
    ])

    support = torch.cat([support, even_curr], axis=1)
    return support


def sample_support_dim5(eq, curr_p, cfg):
    sym = []
    if not eq.support:
        distribution = torch.distributions.Uniform(cfg.fun_support.min,cfg.fun_support.max) #torch.Uniform.distribution_support(cfg.fun_support[0],cfg.fun_support[1])
    else:
        raise NotImplementedError
    
    assert len(cfg.total_variables) == 5

    for sy in cfg.total_variables:
        if sy in eq.variables:
            curr = distribution.sample([int(curr_p)])
        else:
            curr = torch.zeros(int(curr_p))
        sym.append(curr)
    support = torch.stack(sym)

    _even_curr = torch.linspace(cfg.fun_support.min,cfg.fun_support.max,4)
    _even_curr_x, _even_curr_y, _even_curr_z, _even_curr_w, _even_curr_u = torch.meshgrid(_even_curr, _even_curr, _even_curr, _even_curr, _even_curr)

    even_curr = torch.stack([
        _even_curr_x.flatten(),
        _even_curr_y.flatten(),
        _even_curr_z.flatten(),
        _even_curr_w.flatten(),
        _even_curr_u.flatten()
    ])

    support = torch.cat([support, even_curr], axis=1)
    return support

def sample_constants(eq, curr_p, cfg):
    consts = []
    #eq_c = set(eq.coeff_dict.values())
    for c in cfg.total_coefficients:
        if c[:2] == "cm":
            if c in eq.coeff_dict:
                curr = torch.ones([int(curr_p)]) * eq.coeff_dict[c]
            else:
                curr = torch.ones([int(curr_p)])
        elif c[:2] == "ca":
            if c in eq.coeff_dict:
                curr = torch.ones([int(curr_p)]) * eq.coeff_dict[c]
            else:
                curr = torch.zeros([int(curr_p)])
        consts.append(curr)
    return torch.stack(consts)


def evaluate_and_wrap(eqs: List[Equation], cfg, DIM, return_n=False):

    vals = []
    cond0 = []
    # tokens_eqs = [eq.tokenized for eq in eqs]
    # tokens_eqs = tokens_padding(tokens_eqs)
    return_eqs = []

    if DIM == 1:
        # curr_p = 256
        # curr_p = 512
        curr_p = 2048
        # curr_p = 128
    else:
        curr_p = 512

    mask_p = number_of_support_points(cfg.max_number_of_points, cfg.type_of_sampling_points, DIM)
    # mask_p = 32
    # mask_p = 64
    # mask_p = 96

    # print("mask_p", mask_p)

    C = []
    min_max = []

    import time
    for eq in eqs:
        # support = sample_support_dsz(eq, curr_p, cfg)
        # try:
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore")

        if DIM == 1:
            support = sample_support_dim1(eq, curr_p, cfg)
            consts = sample_constants(eq, curr_p+256, cfg)            # XXX
        elif DIM == 2:
            support = sample_support_dim2(eq, curr_p, cfg)
            consts = sample_constants(eq, curr_p+256, cfg)            # XXX
        elif DIM == 3:
            support = sample_support_dim3(eq, curr_p, cfg)
            consts = sample_constants(eq, curr_p+512, cfg)            # XXX
        elif DIM == 4:
            support = sample_support_dim4(eq, curr_p, cfg)
            consts = sample_constants(eq, curr_p+625, cfg)            # XXX
        elif DIM == 5:
            support = sample_support_dim5(eq, curr_p, cfg)
            consts = sample_constants(eq, curr_p+1024, cfg)            # XXX
        else:
            raise NotImplementedError

        # CRITICAL
        # critical = find_critical_dim_1(eq, consts[:,0:1], cfg)
        # # print(critical)
        # # input()

        # if len(critical) == 0:
        #     pass
        # else:
        #     # pdb.set_trace()
        #     support[0, 0:critical.size(0)] = critical
        #     mask_p = max(mask_p, critical.size(0))

        # # pdb.set_trace()
        # # CRITICAL
        # x_critical, y_critical, z_y_critical = find_critical_dim_2(eq, consts[:,0:1], cfg)
        # # pdb.set_trace()
        # # print(critical)
        # # input()

        # if len(x_critical) == 0:
        #     pass
        # else:
        #     # pdb.set_trace()
        #     # support[0, 0:x_critical.size(0)] = x_critical
        #     # support[1, 0:x_critical.size(0)] = y_critical
        #     mask_p = max(mask_p, x_critical.size(0))

        input_lambdi = torch.cat([support, consts], axis=0)
        C.append(consts)

        aaaa = eq.code(*input_lambdi)
        # pdb.set_trace()

        if type(aaaa) == torch.Tensor and aaaa.dtype == torch.float32:
            vals.append(
                torch.cat(
                    [support, torch.unsqueeze(aaaa, axis=0)], axis=0
                ).unsqueeze(0)
            )
            cond0.append(True)
        else:
            cond0.append(False)
        # except NameError as e:
        #     # print(e)
        #     cond0.append(False)
        # except RuntimeError as e:
        #     cond0.append(False)
        # except:
        #     breakpoint()

    # tokens_eqs = tokens_eqs[cond0]
    return_eqs = [e for ii, e in enumerate(eqs) if cond0[ii]]
    min_max = [m for ii, m in enumerate(min_max) if cond0[ii]]

    num_tensors = torch.cat(vals, axis=0)
    cond = (
        torch.sum(torch.count_nonzero(torch.isnan(num_tensors), dim=2), dim=1)
        < 1
    )
    num_fil_nan = num_tensors[cond]
    # tokens_eqs = tokens_eqs[cond]
    return_eqs = [e for ii, e in enumerate(return_eqs) if cond[ii]]
    min_max = [m for ii, m in enumerate(min_max) if cond[ii]]

    cond2 = (
        torch.sum(
            torch.count_nonzero(torch.abs(num_fil_nan) > 5e4, dim=2), dim=1
        )  # Luca comment 0n 21/01
        < 1
    )
    num_fil_nan_big = num_fil_nan[cond2]
    # tokens_eqs = tokens_eqs[cond2]
    return_eqs = [e for ii, e in enumerate(return_eqs) if cond2[ii]]
    min_max = [m for ii, m in enumerate(min_max) if cond2[ii]]

    res = num_fil_nan_big

    # idx = torch.argsort(num_fil_nan_big[:, -1, :]).unsqueeze(1).repeat(1, num_fil_nan_big.shape[1], 1)
    # assert not torch.any(torch.isnan(res))
    # assert not torch.any(torch.abs(res) > 5e4)
    # pdb.set_trace()

    # res = torch.gather(num_fil_nan_big, 2, idx)
    # # res, _ = torch.sort(num_fil_nan_big)
    # res = res[:, :, torch.sum(torch.count_nonzero(torch.isnan(res), dim=1), dim=0) == 0]
    # res = res[
    #     :,
    #     :,
    #     torch.sum(torch.count_nonzero(torch.abs(res) > 5e4, dim=1), dim=0)
    #     == 0,  # Luca comment 0n 21/01
    # ]

    # shuffle_idx = [torch.randperm(res.size(2)) for i in range(res.size(0))]
    # shuffle_idx = torch.stack(shuffle_idx).unsqueeze(1).repeat(1, res.size(1), 1)
    # res = torch.gather(res, 2, shuffle_idx)

    # normalize y
    # pdb.set_trace()

    d_max = res.size(1) - 1

    y_max, _ = res[:, d_max:, :].max(axis=-1)
    y_max = y_max.unsqueeze(-1)
    y_min, _ = res[:, d_max:, :].min(axis=-1)
    y_min = y_min.unsqueeze(-1)

    if DIM == 1:
        res = res[:, :, :-256]
    elif DIM == 2:
        res = res[:, :, :-256]
    elif DIM == 3:
        res = res[:, :, :-512]
    elif DIM == 4:
        res = res[:, :, :-625]
    elif DIM == 5:
        res = res[:, :, :-1024]
    else:
        raise NotImplementedError

    _s = (0.9 + random.random() * 0.1)
    res[:, d_max:, :] = (res[:, d_max:, :] - y_min) / (y_max - y_min + 0.01) * _s

    # _s = (0.9 + random.random() * 0.1)
    # res[:, d_max:, :] = (res[:, d_max:, :] - (y_max+y_min)*0.5) / (y_max - y_min + 0.01) * 2.0 * _s
    # res[:, d_max:, :] = ((res[:, d_max:, :] - y_min) / (y_max - y_min + 1e-5)  * (1.0 - (-1.0)) + (-1.0) ) * _s
    # res[:, d_max:, :] = ((res[:, d_max:, :] - y_min) / (y_max - y_min + 1e-5)) * _s

    assert not torch.any(torch.isnan(res))

    if False:
        noise = torch.normal(0.0, 0.05, size=res[:, d_max:, :].shape)
        res[:, d_max:, :] += noise

    res = res.transpose(1,2)

    given_x, given_y, inter_x, inter_y = res[:, :mask_p, :DIM], res[:, :mask_p, DIM:], res[:, mask_p:, :DIM], res[:, mask_p:, DIM:]

    return_eq_expr = [eq.expr_dsz for eq in return_eqs]

    if return_n:
        return given_x, R_log(given_y), inter_x, R_log(inter_y), (return_eq_expr, (y_min, y_max, _s))
    else:
        return given_x, R_log(given_y), inter_x, R_log(inter_y), mask_p


class NeSymReSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        train_path,
        test_path,
        batch_size,
        num_workers,
        nesymres_data_cfg
    ):
        super().__init__()
        self.data_train_path = os.path.join(data_root, train_path)
        self.data_val_path = os.path.join(data_root, test_path)
        self.data_test_path = os.path.join(data_root, test_path)

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.nesymres_data_cfg = nesymres_data_cfg


    def setup(self, stage=None):
        """called one ecah GPU separately - stage defines if we are at fit or test step"""
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == "fit" or stage is None:

            if self.data_train_path:
                self.training_dataset = NesymresDataset(
                    self.data_train_path,
                    self.nesymres_data_cfg.dataset_train,
                    mode="train"
                )
            
            if self.data_val_path:
                self.validation_dataset = NesymresDataset(
                    self.data_val_path,
                    self.nesymres_data_cfg.dataset_val,
                    mode="val"
                )
            
            if self.data_test_path:
                self.test_dataset = NesymresDataset(
                    self.data_test_path, self.nesymres_data_cfg.dataset_test,
                    mode="test"
                )

    def train_dataloader(self):
        """returns training dataloader"""
        trainloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=partial(custom_collate_fn,cfg= self.nesymres_data_cfg.dataset_train, DIM=len(self.training_dataset.cfg.total_variables)),
            num_workers=self.num_workers,
            pin_memory=True
        )
        return trainloader

    def val_dataloader(self):
        """returns validation dataloader"""
        validloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=partial(custom_collate_fn,cfg= self.nesymres_data_cfg.dataset_val, DIM=len(self.validation_dataset.cfg.total_variables)),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        return validloader

    def test_dataloader(self):
        """returns validation dataloader"""
        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=partial(custom_collate_fn,cfg=self.nesymres_data_cfg.dataset_test, DIM=len(self.test_dataset.cfg.total_variables), return_n=True),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

        return testloader


# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

class NesymresTestDataset(data.Dataset):
    def __init__(self, _dataset):
        #m = Manager()
        self.data = _dataset

        # for batch_id, batch in enumerate(_dataset):
        #     ___batch_size = batch[0].size(0)
        #     for ii in range(___batch_size):
        #         self.data.append(
        #             (batch[0][ii], batch[1][ii], batch[2][ii], batch[3][ii], batch[4])
        #         )

        # pdb.set_trace()

        self.len = len(self.data)


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def saved_test_data(dataset_path, return_loader=True):

    import pickle
    with open(dataset_path, "rb") as f:
        _dataset = pickle.load(f)

    dataset = NesymresTestDataset(_dataset)

    if return_loader:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False
        )

        return dataloader
    else:
        return dataset
