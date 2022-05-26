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
from src.data.NeSymReS_data import saved_test_data

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

    assert hparams.dataset_type == "NeSymReS"

    model = Model(
        hparams,
        DefaultMunch.fromDict(eval(hparams.model_arch_cfg)),
        default_layout=None,
    )

    print( pl.core.memory.ModelSummary(model, mode="full") )

    print(hparams.resume_from_checkpoint)
    print(hparams.resume_from_checkpoint)
    print(hparams.resume_from_checkpoint)
    print(hparams.resume_from_checkpoint)

    # assert hparams.resume_from_checkpoint
    print("Load Pre-Trained Model From", hparams.resume_from_checkpoint, "...")
    checkpoint = torch.load(hparams.resume_from_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------


    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpu,
        precision=16 if hparams.use_16bit else 32,
        profiler=hparams.profiler,
    )

    testdata_loader = saved_test_data(hparams.data_root+"_test_saved.pkl")

    # pdb.set_trace()

    # ------------------------
    # 3 START TESTING
    # ------------------------
    print(hparams)
    print()
    # trainer.fit(model, data)
    trainer.test(model, testdata_loader)

    # trainer.test(model, data.val_dataloader())
    # trainer.test()
