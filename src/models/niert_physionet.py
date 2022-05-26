import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pdb

optimize_mask = False
# optimize_mask = False

NIERT_DIAG = False


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, dim_Inner, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)

        self.fc_o_0 = nn.Linear(dim_V, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_Inner)
        self.fc_o_2 = nn.Linear(dim_Inner, dim_V)

        # self.dropout_1 = nn.Dropout(dropout)
        # self.dropout_2 = nn.Dropout(dropout)

    
    def forward(self, Q, K, mask_pos=None):
        forward = self.forward_noop
        return forward(Q, K, mask_pos)

        # import time
        # start = time.time()
        # for i in range(100):
        #     x = self.forward_op(Q, K, mask_pos)
        # print(time.time() - start)


        # start = time.time()
        # for i in range(100):
        #     y = self.forward_noop(Q, K, mask_pos)
        # print(time.time() - start)

        # pdb.set_trace()


    def forward_noop(self, Q, K, mask_pos=None):

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # pdb.set_trace()
        A_logits = Q_.bmm(K_.transpose(1,2)) / math.sqrt(self.dim_V)


        if mask_pos is not None:
            # mask_ = torch.zeros(Q.size(0), A_logits.size(1), A_logits.size(2), device=A_logits.device)

            mask_pos.masked_fill_(mask_pos.bool(), -math.inf)
            mask_ = mask_pos.unsqueeze(1)

            mask_ = mask_.repeat(self.num_heads,1,1,1)
            mask_ = mask_.reshape(self.num_heads*mask_.size(1), mask_.size(2), mask_.size(3))

            # mask_[:, mask_pos:] = -math.inf

            A_logits += mask_

        A = torch.softmax(A_logits, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + self.fc_o_2(F.relu(self.fc_o(O)))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, d_inner, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, d_inner, num_heads, ln=ln)

    def forward(self, X, mask_pos=None):
        return self.mab(X, X, mask_pos=mask_pos)


class MLPWeight(nn.Module):
    def __init__(self, dim_in, dim_hidden, d_inner, dim_out):
        super(MLPWeight, self).__init__()

        self.linear0 = nn.Linear(dim_in, dim_hidden)
        self.linear1 = nn.Linear(dim_hidden, dim_hidden)            # dddddddddddddddddd
        self.linear2 = nn.Linear(dim_hidden, dim_out)


    # def forward(self, Q, K):
    def forward(self, in_v):

        # QK_cat = Q.unsqueeze(2) + K.unsqueeze(1)

        in_v = F.gelu(self.linear0(in_v))
        in_v = F.gelu(self.linear1(in_v))               # dddddddddddddddddd
        in_v = self.linear2(in_v)

        # pdb.set_trace()

        return in_v.squeeze()

import numpy as np

def subsample_timepoints(data, time_steps, mask, percentage_tp_to_sample=None):
    # Subsample percentage of points from each time series

    inter_tp_mask = mask[:,:,0].clone()
    inter_tp_mask[...] = 0.0

    for i in range(data.size(0)):

        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu()
        non_missing_tp = np.where(current_mask > 0)[0]
        n_tp_current = len(non_missing_tp)
        n_to_sample = int(n_tp_current * percentage_tp_to_sample)

        # print(current_mask)


        subsampled_idx = sorted(np.random.choice(
            non_missing_tp, n_to_sample, replace=False))

        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

        data[i, tp_to_set_to_zero] = 0.

        inter_tp_mask[i, tp_to_set_to_zero] = 1.0

        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask, inter_tp_mask


class NIERT_PhysioNet(nn.Module):
    def __init__ (
        self,
        cfg_dim_input=42,
        cfg_dim_output=41,
        d_model=128,
        # d_model=512,          # fuck!!!!
        d_inner=512,
        n_layers=2,
        # n_head=8,
        n_head=4,
        d_k=128,
        d_v=128,
        SAMPLE_TP=math.nan
    ):
        super().__init__()

        self.dim = 41

        # self.cfg = cfg

        # self.linear = cfg.linear
        # self.bit16 = cfg.bit16
        # self.norm = cfg.norm
        # assert cfg.linear != cfg.bit16, "one and only one between linear and bit16 must be true at the same time" 

        # if cfg.norm:
        # self.register_buffer("mean", torch.tensor(0.5))
        # self.register_buffer("std", torch.tensor(0.5))

        # self.activation = cfg.activation
        # self.input_normalization = cfg.input_normalization
        # if cfg.linear:
            # self.linearl = nn.Linear(cfg_dim_input,16*cfg_dim_input)
        self.linear_a = nn.Parameter(torch.zeros(self.dim, 16))
        self.linear_b = nn.Parameter(torch.zeros(self.dim, 16))

        # if not use_bits:
        self.lineart = nn.Linear(1,16)

            # self.linearr = nn.Linear(cfg_dim_input-1,32)                    # XXX
        
        # self.mask_embedding = nn.Parameter(torch.zeros(16))
        self.mask_embedding = nn.Parameter(torch.zeros(self.dim, 16))

        self.selfatt = nn.ModuleList()
        #dim_input = 32*dim_input
        self.selfatt1 = SAB(16*cfg_dim_input, d_model, d_inner, n_head, ln=True)

        for i in range(n_layers):
            self.selfatt.append(SAB(d_model, d_model, d_inner, n_head, ln=True))

        # self.outatt = MLPWeight(d_model, d_model // 2, 1)
        # an error !!!!
        self.outatt = MLPWeight(d_model, d_model, d_model, cfg_dim_output)

        # self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()

        self.SAMPLE_TP = SAMPLE_TP


    def forward(self, train_batch):

        # train_batch, iii = train_batch

        batch_len = train_batch.shape[0]

        # dim = train_batch.shape[-1] // 2

        observed_data = train_batch[:, :, :self.dim]
        observed_mask = train_batch[:, :, self.dim:2 * self.dim]
        observed_tp = train_batch[:, :, -1]

        subsampled_data, subsampled_tp, subsampled_mask, inter_tp_mask = subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), self.SAMPLE_TP)


        x_emb = subsampled_data.unsqueeze(-1) * self.linear_a.unsqueeze(0).unsqueeze(0) + self.linear_b.unsqueeze(0).unsqueeze(0)
        x_emb = x_emb * subsampled_mask.unsqueeze(-1) + self.mask_embedding.unsqueeze(0).unsqueeze(0) * (1.0 - subsampled_mask.unsqueeze(-1))
        t_emb = self.lineart(subsampled_tp.unsqueeze(-1))


        x_emb = x_emb.reshape((x_emb.size(0), x_emb.size(1), -1))
        xt_emb = torch.cat([x_emb, t_emb], axis=2)

        subsampled_data

        xt_emb = self.selfatt1(xt_emb, mask_pos=inter_tp_mask.clone())

        for layer in self.selfatt:
            xt_emb = layer(xt_emb, mask_pos=inter_tp_mask.clone())

        predict_x = self.outatt(xt_emb)

        # return predict_y, target_y, mask_pos
        # return predict_y.unsqueeze(-1), label_target_y.unsqueeze(-1)
        return predict_x, observed_data, (observed_mask, inter_tp_mask)
