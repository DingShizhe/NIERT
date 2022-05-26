import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pdb

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

    def forward(self, Q, K, mask_pos=None):
        forward = self.forward_op
        return forward(Q, K, mask_pos)

    def forward_op(self, Q, K, mask_pos=None):

        self.fuck_mask = None
        assert mask_pos is not None

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        if False:
            # given, masked, all
            K_given = K_[:, :mask_pos, :]
            A_logits_given = Q_.bmm(K_given.transpose(1,2))/math.sqrt(self.dim_V)

            Q_masked, K_masked = Q_[:, mask_pos:, :], K_[:, mask_pos:, :]
            fb_size, p_num, e_size = Q_masked.shape
            Q_masked, K_masked = Q_masked.reshape((-1, 1, e_size)), K_masked.reshape((-1, e_size, 1))

            A_given_given, A_masked_given = A_logits_given[:, :mask_pos,:], A_logits_given[:, mask_pos:,:]
            A_given_given = torch.softmax(A_given_given, 2)

            V_given  = A_given_given.bmm(V_[:, :mask_pos, :])

            # if NIERT_DIAG:
            if False:
                A_logits_masked_diag = Q_masked.bmm(K_masked).reshape((fb_size, p_num, 1)) / math.sqrt(self.dim_V)
                A_masked_ = torch.cat([A_masked_given, A_logits_masked_diag], axis=-1)
                A_masked_ = torch.softmax(A_masked_, 2)
                V_masked = A_masked_[:,:,:-1].bmm(V_[:, :mask_pos, :]) + A_masked_[:,:,-1:] * V_[:, mask_pos:, :]
            else:
                A_masked_ = A_masked_given
                A_masked_ = torch.softmax(A_masked_, 2)
                V_masked = A_masked_.bmm(V_[:, :mask_pos, :])

            # self.fuck_attn = A_masked_
            # A_masked_ = self.dropout_2(A_masked_)

            V__ = torch.cat([V_given, V_masked], axis=1)

            V__ = torch.cat((V__).split(Q.size(0), 0), 2)
            V__ = self.fc_o_0(V__)
            # pdb.set_trace()

            O = Q + V__

        # V__ = self.dropout_1(V__)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)

        fc_O = self.fc_o_2(F.relu(self.fc_o(O)))
        # fc_O = self.dropout_2(fc_O)
        O = O + fc_O
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


class NIERT(nn.Module):
    def __init__ (
        self,
        cfg_dim_input=3,
        cfg_dim_output=-1,
        d_model=128,
        # d_model=512,          # fuck!!!!
        d_inner=512,
        n_layers=2,
        # n_head=8,
        n_head=4,
        d_k=128,
        d_v=128,
    ):
        super().__init__()

        self.register_buffer("mean", torch.tensor(0.5))
        self.register_buffer("std", torch.tensor(0.5))


        self.linearl = nn.Linear(cfg_dim_input-1,16*(cfg_dim_input-1))
        self.linearr = nn.Linear(1,16)

        self.mask_embedding = nn.Parameter(torch.zeros(16))

        self.selfatt = nn.ModuleList()
        self.selfatt1 = SAB(16*cfg_dim_input, d_model, d_inner, n_head, ln=True)

        for i in range(n_layers):
            self.selfatt.append(SAB(d_model, d_model, d_inner, n_head, ln=True))

        self.outatt = MLPWeight(d_model, d_model, d_model, 1)


    def forward(self, obs_index, heat_obs, pred_index, heat):

        mask_pos = obs_index.size(1)

        label_y = torch.cat([heat_obs, heat], axis=1).squeeze()

        given_x, given_y = self.linearl(obs_index), self.linearr(heat_obs)
        given_xy = torch.cat((given_x, given_y), dim=-1)

        target_x = self.linearl(pred_index)
        target_y = self.mask_embedding.view((1,1,16)).expand((target_x.size(0), target_x.size(1), 16))
        target_xy = torch.cat((target_x, target_y), dim=-1)
        given_target_xy = torch.cat((given_xy, target_xy), dim=1)

        given_target_xy = self.selfatt1(given_target_xy, mask_pos=mask_pos)

        for layer in self.selfatt:
            given_target_xy = layer(given_target_xy, mask_pos=mask_pos)

        predict_y = self.outatt(given_target_xy)

        # return predict_y, target_y, mask_pos
        return predict_y.unsqueeze(-1), label_y.unsqueeze(-1)
