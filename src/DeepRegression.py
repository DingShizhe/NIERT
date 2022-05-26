# encoding: utf-8
import imp
import math
from pathlib import Path
from typing import SupportsAbs

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
import numpy as np

from src.data.TFR_data import LayoutDataset, LayoutVecDataset
import src.models as models
import pdb


# for tfr finetune
finetune_mean_factor = (0.34 / 0.094)


def mean_squared_error(orig, pred, mask):
    # pdb.set_trace()
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def mean_squared_error_inter(orig, pred, mask, inter_mask):
    # pdb.set_trace()
    error = (orig - pred) ** 2
    _mask = mask * inter_mask.unsqueeze(-1)
    error = error * _mask
    return error.sum() / _mask.sum()


class Model(LightningModule):
    def __init__(self, hparams, model_arch_cfg, default_layout=None):
        super().__init__()
        self.hparams = hparams
        self.model_arch_cfg = model_arch_cfg
        self._build_model()

        if hparams.dataset_type in ["NeSymReS"]:
            self.criterion = nn.MSELoss()
        elif hparams.dataset_type in ["TFR", "TFR_FINETUNE"]:
            self.criterion = nn.L1Loss()
        elif hparams.dataset_type in ["PhysioNet", "NeSymReS_42", "PhysioNet_FINETUNE"]:
            self.criterion = mean_squared_error
        elif hparams.dataset_type in ["Perlin"]:
            self.criterion = nn.MSELoss()
        elif hparams.dataset_type in ["Current"]:
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError

        if hparams.dataset_type == "TFR_FINETUNE":
            self._finetune_mean_factor = finetune_mean_factor
        else:
            self._finetune_mean_factor = 1.0

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.default_layout = default_layout


    def _build_model(self):
        model_list = [
            "TransformerRecon",
            "NIERT",
            "NIERT_PhysioNet",
            "ConditionalNeuralProcess",
            "AttentiveNeuralProcess",
            "NIERTPP"
        ]
        self.layout_model = self.hparams.model_name

        if self.hparams.dataset_type in ["PhysioNet", "NeSymReS_42", "PhysioNet_FINETUNE"]:
            assert self.layout_model == "NIERT_PhysioNet"
        else:
            assert not self.layout_model == "NIERT_PhysioNet"

        assert self.layout_model in model_list, "Error: Model {self.layout_model} Not Defined"


        model_args = dict(
            cfg_dim_input=self.hparams.cfg_dim_input,
            cfg_dim_output=self.hparams.cfg_dim_output,
            n_layers=self.model_arch_cfg.layers,
            d_model=self.model_arch_cfg.dim_hidden,
            d_inner=self.model_arch_cfg.dim_inner,
            n_head=self.model_arch_cfg.num_heads
        )

        if self.hparams.dataset_type in ["PhysioNet", "PhysioNet_FINETUNE"]:
            model_args["SAMPLE_TP"] = self.hparams.SAMPLE_TP
        if self.hparams.model_name == "NIERTPP":
            model_args["n_inds"] = _head=self.model_arch_cfg.num_inds

        self.model = getattr(models, self.layout_model) (**model_args)


    def forward(self, x):

        # Data difference betteen TFR and NeSymReS
        masks = None

        if self.hparams.dataset_type == "PhysioNet_FINETUNE":
            x = x.clone()
            x[:,:,-1] = (x[:,:,-1]  / 0.268) - 1.0
            # self.log("train/mean_tp", x[:, :, -1].mean())
            output, label, masks = self.model( x )

        elif self.hparams.dataset_type in ["NeSymReS_42"]:
            x = x.clone()
            # x[:,:,:41] = x[:,:,:41] / for_finetune_physionet_mean_factor.unsqueeze(0).unsqueeze(0).to(x.device)      # to meet the distribution of xs of  physionet_data !!
            output, label, masks = self.model( x )

        elif self.hparams.dataset_type in ["PhysioNet"]:
            output, label, masks = self.model( x )

        elif self.hparams.dataset_type == "TFR_FINETUNE":

            def ch_range(x): return x * 2.0 - 1.0

            output, label = self.model(
                ch_range(x[0]), x[1] * self._finetune_mean_factor,
                ch_range(x[2]), x[3] * self._finetune_mean_factor
            )

        elif self.hparams.dataset_type in ["NeSymReS", "TFR", "Perlin", "Current"]:
            # import time
            # t = time.time()

            # Rebattal
            if False:
                noise = torch.normal(0.0, 0.02, size=x[1].shape, device=x[1].device)
                x[1] += noise

                # noise = torch.normal(0.0, 0.05, size=x[2].shape, device=x[2].device)
                # x[2] += noise

            # opn = x[0].size(1)
            # tpn = x[2].size(1)
            # outputs = []

            # _p = opn // 10

            # for i in range(10):

            #     x_0 = torch.cat([x[0][:, :_p*i ,:], x[0][:, _p*(i+1): ,:]], dim=1)
            #     x_1 = torch.cat([x[1][:, :_p*i ,:], x[1][:, _p*(i+1): ,:]], dim=1)

            #     output_i, _ = self.model( x_0, x_1, x[2], x[3] )
            #     outputs.append(output_i[:,-tpn:,:])

            # output = sum(outputs) / 10.0

            # output = torch.cat([ x[1] , output], dim=1)
            # label = torch.cat([ x[1] , x[3]], dim=1)
            # pdb.set_trace()

            if False:
                RRR = 32

                output, label = self.model( x[0], x[1], x[2], x[3] )
                output_0, label_0 = self.model( x[0], x[1], x[2][:,0:RRR,:], x[3][:,0:RRR,:] )

            # pdb.set_trace()
            # print(time.time() - t, "sec...")
            else:
                output, label = self.model( x[0], x[1], x[2], x[3] )

        else:
            raise NotImplementedError

        if masks is not None:
            return output, label, masks
        else:
            # return output, label, output_0, label_0
            return output, label

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        # if self.hparams.lr_decay < 0.0:
        if self.hparams.dataset_type == "NeSymReS":
            return optimizer
        elif self.hparams.dataset_type in ["TFR", "TFR_FINETUNE"]:
            scheduler = ExponentialLR(optimizer, gamma=self.hparams.lr_decay)
            return [optimizer], [scheduler]
        elif self.hparams.dataset_type in ["PhysioNet", "PhysioNet_FINETUNE"]:
            return optimizer
        elif self.hparams.dataset_type == "NeSymReS_42":
            return optimizer
        elif self.hparams.dataset_type == "Perlin":
            return optimizer
        elif self.hparams.dataset_type == "Current":
            return optimizer
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx):

        if self.layout_model == "NIERT_PhysioNet":
            heat_pred, heat_label, masks = self(batch)
        else:
            obs_index, heat_obs, pred_index, heat, _ = batch
            heat_info = [obs_index, heat_obs, pred_index, heat]

            heat_pred, heat_label = self(heat_info)

        # self.log("train/batch_idx", batch_idx)
        # self.log("train/heat_label", heat_label.mean() / self._finetune_mean_factor)
        # self.log("train/heat_pred", heat_pred.mean() / self._finetune_mean_factor)

        if self.layout_model in ["NIERT", "NIERTPP"]:
            masked_loss = self.criterion(heat_label[:, obs_index.size(1):, :], heat_pred[:, obs_index.size(1):, :]) * self.hparams.std_heat / self._finetune_mean_factor
            self.log("train/training_mae", masked_loss, on_epoch=True)

            observed_loss = self.criterion(heat_label[:, :obs_index.size(1), :], heat_pred[:, :obs_index.size(1), :]) * self.hparams.std_heat / self._finetune_mean_factor
            self.log("train/training_observed_mae", observed_loss, on_epoch=True)

            niert_loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat / self._finetune_mean_factor
            self.log("train/training_mae_niert", niert_loss, on_epoch=True)

            return {"loss": niert_loss, "masked_loss": masked_loss}

        elif self.layout_model == "NIERT_PhysioNet":
            loss = self.criterion(heat_label, heat_pred, masks[0])
            self.log("train/training_mae", loss, on_epoch=True)

            masked_loss = mean_squared_error_inter(heat_label, heat_pred, masks[0], masks[1])
            self.log("train/training_mae_niert", masked_loss, on_epoch=True)

            return {"loss": loss, "masked_loss": masked_loss}

        else:
            loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat / self._finetune_mean_factor
            self.log("train/training_mae", loss, on_epoch=True)

            return {"loss": loss}

    # def training_epoch_end(self, outputs):
    #     if self.layout_model == "NIERT":
    #         train_loss_mean = torch.stack([x["masked_loss"] for x in outputs]).mean()
    #         self.log("train/train_mae_epoch", train_loss_mean.item())

    #         train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
    #         self.log("train/train_mae_niert_epoch", train_loss_mean.item())
    #     else:
    #         train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
    #         self.log("train/train_mae_epoch", train_loss_mean.item())

    def validation_step(self, batch, batch_idx):

        # pdb.set_trace()

        if self.layout_model == "NIERT_PhysioNet":
            heat_pred, heat_label, masks = self(batch)
        else:
            obs_index, heat_obs, pred_index, heat, _ = batch
            heat_info = [obs_index, heat_obs, pred_index, heat]

            heat_pred, heat_label = self(heat_info)

        if self.layout_model in ["NIERT", "NIERTPP"]:
            masked_loss = self.criterion(heat_label[:, obs_index.size(1):, :], heat_pred[:, obs_index.size(1):, :]) * self.hparams.std_heat / self._finetune_mean_factor
            self.log("val/val_mae", masked_loss, on_epoch=True)

            observed_loss = self.criterion(heat_label[:, :obs_index.size(1), :], heat_pred[:, :obs_index.size(1), :]) * self.hparams.std_heat / self._finetune_mean_factor
            self.log("val/val_observed_mae", observed_loss, on_epoch=True)

            niert_loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat / self._finetune_mean_factor
            self.log("val/val_mae_niert", niert_loss, on_epoch=True)

            return {"val_loss": niert_loss, "val_masked_loss": masked_loss}

        elif self.layout_model == "NIERT_PhysioNet":
            loss = self.criterion(heat_label, heat_pred, masks[0])
            self.log("val/val_mae", loss, on_epoch=True)

            masked_loss = mean_squared_error_inter(heat_label, heat_pred, masks[0], masks[1])
            self.log("val/val_mae_niert", masked_loss, on_epoch=True)

            return {"val_loss": loss, "val_masked_loss": masked_loss}

        else:
            loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat / self._finetune_mean_factor
            self.log("val/val_mae", loss, on_epoch=True)

            return {"val_loss": loss}


    # def validation_epoch_end(self, outputs):
    #     if self.layout_model == "NIERT":
    #         val_loss_mean = torch.stack([x["val_masked_loss"] for x in outputs]).mean()
    #         self.log("val/val_mae_epoch", val_loss_mean.item())

    #         val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
    #         self.log("val/val_mae_niert_epoch", val_loss_mean.item())
    #     else:
    #         val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
    #         self.log("val/val_mae_epoch", val_loss_mean.item())

    def test_step(self, batch, batch_idx):

        if self.layout_model == "NIERT_PhysioNet":
            heat_pred, heat_label, masks = self(batch)
        else:
            if len(batch[0].shape) == 4:
                assert batch[0].size(0) == 1
                obs_index, heat_obs, pred_index, heat, _ = batch
                obs_index, heat_obs, pred_index, heat = obs_index.squeeze(0), heat_obs.squeeze(0), pred_index.squeeze(0), heat.squeeze(0)
            else:
                obs_index, heat_obs, pred_index, heat, _ = batch

            heat_info = [obs_index, heat_obs, pred_index, heat]
            heat_pred, heat_label = self(heat_info)


        if self.hparams.dataset_type in ["TFR", "TFR_FINETUNE"]:

            if self.layout_model in ["NIERT", "NIERTPP"]:
                heat_label = heat_label[:, obs_index.size(1):, :]
                heat_pred = heat_pred[:, obs_index.size(1):, :]

            heat_label = heat_label.reshape(-1, 1, 200, 200)
            heat_pred = heat_pred.reshape(-1, 1, 200, 200)

            # heat_label = heat_label.transpose(2, 3)
            # heat_pred = heat_pred.transpose(2, 3)

            # masked_loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat / self._finetune_mean_factor
            # self.log("val/val_mae", masked_loss, on_epoch=True)

            # niert_loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat / self._finetune_mean_factor
            # self.log("val/val_mae_niert", niert_loss, on_epoch=True)

            # return {"val_loss": niert_loss, "val_masked_loss": masked_loss}

            loss = self.criterion(heat_label, heat_pred) * self.hparams.std_heat / self._finetune_mean_factor

            default_layout = (
                torch.repeat_interleave(
                    self.default_layout, repeats=heat_pred.size(0), dim=0
                )
                .float()
                .to(device=heat_label.device)
            )
            ones = torch.ones_like(default_layout).to(device=heat_label.device)
            zeros = torch.zeros_like(default_layout).to(device=heat_label.device)
            layout_ind = torch.where(default_layout < 1e-2, zeros, ones)
            loss_2 = (
                torch.sum(torch.abs(torch.sub(heat_label, heat_pred)) * layout_ind)
                * self.hparams.std_heat
                / torch.sum(layout_ind)
            ) / self._finetune_mean_factor
            # ---------------------------------
            loss_1 = (
                torch.sum(
                    torch.max(
                        torch.max(
                            torch.max(
                                torch.abs(torch.sub(heat_label, heat_pred)) * layout_ind, 3
                            ).values,
                            2,
                        ).values
                        * self.hparams.std_heat,
                        1,
                    ).values
                )
                / heat_pred.size(0)
            ) / self._finetune_mean_factor
            # ---------------------------------
            boundary_ones = torch.zeros_like(default_layout).to(device=heat_label.device)
            boundary_ones[..., -2:, :] = ones[..., -2:, :]
            boundary_ones[..., :2, :] = ones[..., :2, :]
            boundary_ones[..., :, :2] = ones[..., :, :2]
            boundary_ones[..., :, -2:] = ones[..., :, -2:]
            loss_3 = (
                torch.sum(torch.abs(torch.sub(heat_label, heat_pred)) * boundary_ones)
                * self.hparams.std_heat
                / torch.sum(boundary_ones)
            ) / self._finetune_mean_factor
            # ----------------------------------

            loss_4 = (
                torch.sum(
                    torch.max(
                        torch.max(
                            torch.max(torch.abs(torch.sub(heat_label, heat_pred)), 3).values, 2
                        ).values
                        * self.hparams.std_heat,
                        1,
                    ).values
                )
                / heat_pred.size(0)
            ) / self._finetune_mean_factor

            # pdb.set_trace()

            return {
                "test_loss": loss,
                "test_loss_1": loss_1,
                "test_loss_2": loss_2,
                "test_loss_3": loss_3,
                "test_loss_4": loss_4,
                "task_to_be_saved": (obs_index.cpu(), heat_obs.cpu(), pred_index.cpu(), heat.cpu(), heat_pred.cpu()) if batch_idx < 50 else None
            }

        elif self.hparams.dataset_type in ["PhysioNet", "PhysioNet_FINETUNE"]:
        
            loss = self.criterion(heat_label, heat_pred, masks[0])
            # self.log("val/val_mae", loss, on_epoch=True)

            masked_loss = mean_squared_error_inter(heat_label, heat_pred, masks[0], masks[1])
            # self.log("val/val_mae_niert", masked_loss, on_epoch=True)

            return {
                "val_loss": loss,
                "val_masked_loss": masked_loss,
                "batch_size": heat_label.size(0)
            }

        elif self.hparams.dataset_type in ["NeSymReS", "Perlin", "Current"]:

            test_criterion = torch.nn.MSELoss(reduction='none')

            losses_t = test_criterion(heat_label[:, obs_index.size(1):, :], heat_pred[:, obs_index.size(1):, :])
            losses_o = test_criterion(heat_label[:, :obs_index.size(1), :], heat_pred[:, :obs_index.size(1), :])

            losses = test_criterion(heat_label, heat_pred) * self.hparams.std_heat / self._finetune_mean_factor

            losses = losses.squeeze(-1).mean(axis=-1)

            # self.log("val/val_mae", loss, on_epoch=True)

            # RRR = min(heat_pred.size(1), output_0.size(1)) - obs_index.size(1)
            # print(RRR)

            # print(heat_pred[:,obs_index.size(1):obs_index.size(1)+RRR,:].shape, output_0[:,obs_index.size(1):obs_index.size(1)+RRR,:].shape)

            return {
                "val_loss_observed": losses_o,
                "val_loss_target": losses_t,
                "val_loss": losses,
                # "mean_std_rebuttal": torch.abs(heat_pred[:,0:1,:] - output_0[:,0:1,:]).mean(),
                # "mean_std_rebuttal": torch.abs(heat_pred[:,obs_index.size(1):obs_index.size(1)+RRR,:] - output_0[:,obs_index.size(1):obs_index.size(1)+RRR,:]).mean(),
                "given_points": obs_index.size(1),
            }

        else:
            raise NotImplementedError


    def test_epoch_end(self, outputs):

        if self.hparams.dataset_type in ["TFR", "TFR_FINETUNE"]:
            test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
            self.log("test_loss (" + "MAE" + ")", test_loss_mean.item())
            # test_loss_max = torch.max(torch.stack([x["test_loss_1"] for x in outputs]))
            test_loss_max = torch.stack([x["test_loss_1"] for x in outputs]).mean()
            self.log("test_loss_1 (" + "M-CAE" + ")", test_loss_max.item())
            test_loss_com_mean = torch.stack([x["test_loss_2"] for x in outputs]).mean()
            self.log("test_loss_2 (" + "CMAE" + ")", test_loss_com_mean.item())
            test_loss_bc_mean = torch.stack([x["test_loss_3"] for x in outputs]).mean()
            self.log("test_loss_3 (" + "BMAE" + ")", test_loss_bc_mean.item())
            test_loss_max_1 = torch.stack([x["test_loss_4"] for x in outputs]).mean()
            self.log("test_loss_4 (" + "MaxAE" + ")", test_loss_max_1.item())


            # ONLY TFR?
            task_to_be_saved = [x["task_to_be_saved"] for x in outputs if x["task_to_be_saved"]]

            # pdb.set_trace()

            import pickle
            dump_path = "vis_results/%s_BBBB%s_res.pkl" % (self.hparams.dataset_type, self.hparams.model_name)
            with open(dump_path, "wb") as f:
                pickle.dump(task_to_be_saved, f)
            print("Some res dumped to", dump_path)


        elif self.hparams.dataset_type in ["PhysioNet", "PhysioNet_FINETUNE"]:

            test_loss_sum = torch.stack([x["val_masked_loss"] * x["batch_size"] for x in outputs]).sum()
            test_loss_mean = test_loss_sum / sum([x["batch_size"] for x in outputs])
            # test_loss_mean = torch.stack([x["val_masked_loss"] for x in outputs]).mean()
            self.log("MSE Error", test_loss_mean.item())

            return test_loss_mean

        elif self.hparams.dataset_type in ["NeSymReS", "Perlin", "Current"]:

            points_nums = set([x["given_points"] for x in outputs])
            loss_by_num = {
                pn:[] for pn in points_nums
            }
            for x in outputs:
                loss_by_num[x["given_points"]].append(x["val_loss"])

            _loss_by_num = {
                pn:torch.cat(loss_by_num[pn]).mean().item() for pn in loss_by_num
            }

            test_loss_mean = torch.stack([x["val_loss"].mean() for x in outputs]).mean()
            test_loss_target_mean = torch.stack([x["val_loss_target"].mean() for x in outputs]).mean()
            test_loss_observed_mean = torch.stack([x["val_loss_observed"].mean() for x in outputs]).mean()

            # test_mean_std_rebuttal = torch.stack([x["mean_std_rebuttal"].mean() for x in outputs]).mean()

            print("losses_by_num = ", _loss_by_num)
            print("val_loss = ", test_loss_mean.item())
            print("val_loss_target = ", test_loss_target_mean.item())
            print("val_loss_observed = ", test_loss_observed_mean.item())


            # print("Mean std:", test_mean_std_rebuttal.item())
            # print("Mean std:", test_mean_std_rebuttal.item())
            # print("Mean std:", test_mean_std_rebuttal.item())
            # print("Mean std:", test_mean_std_rebuttal.item())

            self.log("val_loss = ", test_loss_mean.item())


            # task_to_be_saved = [x["task_to_be_saved"] for x in outputs if x["task_to_be_saved"]]

        else:
            raise NotImplementedError




    @staticmethod
    def add_model_specific_args(parser):  # pragma: no-cover
        """Parameters you define here will be available to your model through `self.hparams`."""
        # dataset args
        parser.add_argument(
            "--data_root", type=str, required=True, help="path of dataset"
        )
        parser.add_argument(
            "--train_path", type=str, required=True, help="path of train dataset list"
        )
        parser.add_argument(
            "--train_size",
            default=0.8,
            type=float,
            help="train_size in train_test_split",
        )
        parser.add_argument(
            "--test_path", type=str, required=True, help="path of test dataset list"
        )
        # parser.add_argument("--boundary", type=str, default="rm_wall", help="boundary condition")
        parser.add_argument(
            "--data_format",
            type=str,
            default="mat",
            choices=["mat"],
            help="dataset format",
        )

        # Normalization params
        parser.add_argument("--mean_heat", default=0, type=float)
        parser.add_argument("--std_heat", default=1, type=float)

        # Model params (opt)
        parser.add_argument(
            "--model_name", type=str, default="FCN", help="the name of chosen model"
        )

        return parser
