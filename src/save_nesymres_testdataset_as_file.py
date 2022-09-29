"""
Runs a model on a single node across multiple gpus.
"""
from pathlib import Path

import torch
from torch.backends import cudnn
import configargparse
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.DeepRegression import Model
from src.data.TFR_data import TFRDataModule
from src.data.NeSymReS_data import NeSymReSDataModule, evaluate_and_wrap
# from src.data.Perlin_data import PerlinDataModule

from munch import DefaultMunch

import pdb

def main(hparams):

    """
    Main training routine specific for this project
    """
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

    # ------------------------
    # 1 INIT LIGHTNING MODEL DATA
    # ------------------------

    if hparams.dataset_type in ["TFR", "TFR_FINETUNE"]:
        assert False
        data = TFRDataModule(
            hparams.data_root,
            hparams.train_path,
            hparams.test_path,
            hparams.batch_size,
            hparams.num_workers,
        )
    elif hparams.dataset_type == "NeSymReS":
        data = NeSymReSDataModule(
            hparams.data_root,
            hparams.train_path,
            hparams.test_path,
            50,                         # batch_size
            hparams.num_workers,
            DefaultMunch.fromDict(eval(hparams.nesymres_data_cfg))
        )
    elif hparams.dataset_type == "Perlin":
        data = PerlinDataModule(None, None, None, 32, 4)
    else:
        raise NotImplementedError



    res = []

    for iii in range(100):

        print(iii)

        data.setup()
        _T = data.val_dataloader()


        for batch_id, batched_data in enumerate(_T):
            # print(batch_id, len(batched_data[0]))
            # pdb.set_trace()
            res.append(batched_data)

    # pdb.set_trace()

    import pickle, os

    with open(hparams.data_root+"_test_saved.pkl", "wb") as f:
    # with open(hparams.data_root+"_test_saved_32O_96T.pkl", "wb") as f:
        pickle.dump(res, f)
