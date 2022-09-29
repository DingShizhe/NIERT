from pathlib import Path

import torch
from torch.backends import cudnn
import configargparse
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from src.DeepRegression import Model
from src.data.TFR_data import TFRDataModule
from src.data.NeSymReS_data import NeSymReSDataModule
from src.data.NeSymReS_42_data import NeSymReS42DataModule
from src.data.PhysioNet_data import PhysioNetDataModule
from src.data.D30_data import D30DataModule


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
            hparams.batch_size,
            hparams.num_workers,
            DefaultMunch.fromDict(eval(hparams.nesymres_data_cfg))
        )
    elif hparams.dataset_type == "NeSymReS_42":
        data = NeSymReS42DataModule(
            hparams.data_root,
            hparams.train_path,
            hparams.test_path,
            hparams.batch_size,
            hparams.num_workers,
            DefaultMunch.fromDict(eval(hparams.nesymres_data_cfg))
        )
    elif hparams.dataset_type in ["PhysioNet", "PhysioNet_FINETUNE"]:
        data = PhysioNetDataModule(
            hparams.data_root,
            hparams.train_path,
            hparams.test_path,
            hparams.batch_size,
            hparams.num_workers
        )
    elif hparams.dataset_type in ["Perlin"]:
        data = PerlinDataModule(
            hparams.data_root,
            hparams.train_path,
            hparams.test_path,
            hparams.batch_size,
            hparams.num_workers
        )
    elif hparams.dataset_type in ["Current"]:
        data = CurrentDataModule(
            hparams.data_root,
            hparams.train_path,
            hparams.test_path,
            hparams.batch_size,
            hparams.num_workers
        )
    elif hparams.dataset_type in ["D30"]:
        data = D30DataModule(
            hparams.data_root,
            hparams.train_path,
            hparams.test_path,
            hparams.batch_size,
            hparams.num_workers
        )
    else:
        raise NotImplementedError


    model = Model(
        hparams,
        DefaultMunch.fromDict(eval(hparams.model_arch_cfg))
    )

    print( pl.core.memory.ModelSummary(model, mode="full") )


    # if hparams.dataset_type in ["TFR_FINETUNE", "PhysioNet_FINETUNE", "Current"]:
    if hparams.dataset_type in ["TFR_FINETUNE", "PhysioNet_FINETUNE"]:
        assert hparams.resume_from_checkpoint
        print("Load Pre-Trained Model From", hparams.resume_from_checkpoint, "...")
        checkpoint = torch.load(hparams.resume_from_checkpoint)
        model.load_state_dict(checkpoint["state_dict"])

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------

    if hparams.dataset_type in ["TFR", "TFR_FINETUNE"]:
        wandb_logger = WandbLogger(project="TFR")
    elif hparams.dataset_type == "NeSymReS":
        wandb_logger = WandbLogger(project="NULL")
    elif hparams.dataset_type in ["PhysioNet", "PhysioNet_FINETUNE"]:
        wandb_logger = WandbLogger(project="NIERT-PhysioNet-New")
    elif hparams.dataset_type == "NeSymReS_42":
        wandb_logger = WandbLogger(project="NIERT-PRETRAIN-42")         # Pre-Train for PhysioNet
    elif hparams.dataset_type == "D30":
        wandb_logger = WandbLogger(project="NIERT_D30")
        # wandb_logger = WandbLogger(project="NULL")
    else:
        raise NotImplementedError


    from datetime import datetime
    now = datetime.now()
    now = now.strftime("%m-%d-%Y %H:%M:%S")


    checkpoint_callback = ModelCheckpoint(
        # monitor="train/training_mae_step",
        monitor="val/val_mae",
        # monitor="val/val_mae_niert",
        dirpath="CKPTS/" + now,
        filename="log_"+"-{epoch:03d}-{loss:.5f}",
        # mode="min",
        save_top_k=20
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # print(hparams.resume_from_checkpoint)

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpu,
        precision=16 if hparams.use_16bit else 32,
        val_check_interval=hparams.val_check_interval,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        profiler=hparams.profiler,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    print(hparams)
    print()
    # trainer.fit(model, data)
    trainer.fit(model, data)

    # trainer.test()
