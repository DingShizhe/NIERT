import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
import torch
import torch.nn as nn


# from attrdict import AttrDict

# from models.canp import CANP
# from utils.misc import stack, logmeanexp
# from utils.sampling import sample_with_replacement as SWR, sample_subset

from torch.distributions import Normal

class MultiHeadAttn(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_out, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.fc_q = nn.Linear(dim_q, dim_out, bias=False)
        self.fc_k = nn.Linear(dim_k, dim_out, bias=False)
        self.fc_v = nn.Linear(dim_v, dim_out, bias=False)
        self.fc_out = nn.Linear(dim_out, dim_out)
        self.ln1 = nn.LayerNorm(dim_out)
        self.ln2 = nn.LayerNorm(dim_out)

    def scatter(self, x):
        return torch.cat(x.chunk(self.num_heads, -1), -3)

    def gather(self, x):
        return torch.cat(x.chunk(self.num_heads, -3), -1)

    def attend(self, q, k, v, mask=None):
        q_, k_, v_ = [self.scatter(x) for x in [q, k, v]]
        A_logits = q_ @ k_.transpose(-2, -1) / math.sqrt(self.dim_out)
        if mask is not None:
            mask = mask.bool().to(q.device)
            mask = torch.stack([mask]*q.shape[-2], -2)
            mask = torch.cat([mask]*self.num_heads, -3)
            A = torch.softmax(A_logits.masked_fill(mask, -float('inf')), -1)
            A = A.masked_fill(torch.isnan(A), 0.0)
        else:
            A = torch.softmax(A_logits, -1)
        return self.gather(A @ v_)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        out = self.ln1(q + self.attend(q, k, v, mask=mask))
        out = self.ln2(out + F.relu(self.fc_out(out)))
        return out

class SelfAttn(MultiHeadAttn):
    def __init__(self, dim_in, dim_out, num_heads=8):
        super().__init__(dim_in, dim_in, dim_in, dim_out, num_heads)

    def forward(self, x, mask=None):
        return super().forward(x, x, x, mask=mask)



def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth-2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)

class PoolingEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1,
            dim_hid=128, dim_lat=None, self_attn=False,
            pre_depth=4, post_depth=2):
        super().__init__()

        self.use_lat = dim_lat is not None

        self.net_pre = build_mlp(dim_x+dim_y, dim_hid, dim_hid, pre_depth) \
                if not self_attn else \
                nn.Sequential(
                        build_mlp(dim_x+dim_y, dim_hid, dim_hid, pre_depth-2),
                        nn.ReLU(True),
                        SelfAttn(dim_hid, dim_hid))

        self.net_post = build_mlp(dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid,
                post_depth)

    def forward(self, xc, yc, mask=None):
        out = self.net_pre(torch.cat([xc, yc], -1))
        if mask is None:
            out = out.mean(-2)
        else:
            mask = mask.to(xc.device)
            out = (out * mask.unsqueeze(-1)).sum(-2) / \
                    (mask.sum(-1, keepdim=True).detach() + 1e-5)
        if self.use_lat:
            mu, sigma = self.net_post(out).chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return self.net_post(out)

class CrossAttnEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
            dim_lat=None, self_attn=True,
            v_depth=4, qk_depth=2):
        super().__init__()
        self.use_lat = dim_lat is not None

        if not self_attn:
            self.net_v = build_mlp(dim_x+dim_y, dim_hid, dim_hid, v_depth)
        else:
            self.net_v = build_mlp(dim_x+dim_y, dim_hid, dim_hid, v_depth-2)
            self.self_attn = SelfAttn(dim_hid, dim_hid)

        self.net_qk = build_mlp(dim_x, dim_hid, dim_hid, qk_depth)

        self.attn = MultiHeadAttn(dim_hid, dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid)

    def forward(self, xc, yc, xt, mask=None):
        q, k = self.net_qk(xt), self.net_qk(xc)
        v = self.net_v(torch.cat([xc, yc], -1))

        if hasattr(self, 'self_attn'):
            v = self.self_attn(v, mask=mask)

        out = self.attn(q, k, v, mask=mask)
        if self.use_lat:
            mu, sigma = out.chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return out

class Decoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1,
            dim_enc=128, dim_hid=128, depth=3):
        super().__init__()
        self.fc = nn.Linear(dim_x+dim_enc, dim_hid)
        self.dim_hid = dim_hid

        modules = [nn.ReLU(True)]
        for _ in range(depth-2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dim_hid, 2*dim_y))
        self.mlp = nn.Sequential(*modules)

    def add_ctx(self, dim_ctx):
        self.dim_ctx = dim_ctx
        self.fc_ctx = nn.Linear(dim_ctx, self.dim_hid, bias=False)

    def forward(self, encoded, x, ctx=None):
        packed = torch.cat([encoded, x], -1)
        hid = self.fc(packed)
        if ctx is not None:
            hid = hid + self.fc_ctx(ctx)
        out = self.mlp(hid)
        mu, sigma = out.chunk(2, -1)
        # sigma = 0.1 + 0.9 * F.softplus(sigma)
        _sigma = torch.ones_like(sigma, device=sigma.device)
        return Normal(mu, _sigma)


class CANP(nn.Module):
    def __init__(self,
            dim_x=1,
            dim_y=1,
            dim_hid=128,
            enc_v_depth=4,
            enc_qk_depth=2,
            enc_pre_depth=4,
            enc_post_depth=2,
            dec_depth=3):

        super().__init__()

        self.enc1 = CrossAttnEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                v_depth=enc_v_depth,
                qk_depth=enc_qk_depth)

        self.enc2 = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                self_attn=True,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.dec = Decoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_enc=2*dim_hid,
                dim_hid=dim_hid,
                depth=dec_depth)

    def predict(self, xc, yc, xt, num_samples=None):
        assert False
        # theta1 = self.enc1(xc, yc, xt)
        # theta2 = self.enc2(xc, yc)
        # encoded = torch.cat([theta1,
        #     torch.stack([theta2]*xt.shape[-2], -2)], -1)
        # return self.dec(encoded, xt)

    def forward(self, batch, num_samples=None, reduce_ll=True):
        assert False
        # outs = AttrDict()
        # py = self.predict(batch.xc, batch.yc, batch.x)
        # ll = py.log_prob(batch.y).sum(-1)

        # if self.training:
        #     outs.loss = -ll.mean()
        # else:
        #     num_ctx = batch.xc.shape[-2]
        #     if reduce_ll:
        #         outs.ctx_ll = ll[...,:num_ctx].mean()
        #         outs.tar_ll = ll[...,num_ctx:].mean()
        #     else:
        #         outs.ctx_ll = ll[...,:num_ctx]
        #         outs.tar_ll = ll[...,num_ctx:]

        # return outs


def gather(items, idxs):
    K = idxs.shape[0]
    idxs = idxs.to(items[0].device)
    gathered = []
    for item in items:
        gathered.append(torch.gather(
            torch.stack([item]*K), -2,
            torch.stack([idxs]*item.shape[-1], -1)).squeeze(0))
    return gathered[0] if len(gathered) == 1 else gathered

def sample_with_replacement(*items, num_samples=None, r_N=1.0, N_s=None):
    K = num_samples or 1
    N = items[0].shape[-2]
    N_s = N_s or max(1, int(r_N * N))
    batch_shape = items[0].shape[:-2]
    idxs = torch.randint(N, size=(K,)+batch_shape+(N_s,))
    return gather(items, idxs)

SWR = sample_with_replacement


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])

def stack(x, num_samples=None, dim=0):
    return x if num_samples is None \
            else torch.stack([x]*num_samples, dim=dim)



class BANP(CANP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dec.add_ctx(2*kwargs['dim_hid'])

    def encode(self, xc, yc, xt, mask=None):
        theta1 = self.enc1(xc, yc, xt)
        theta2 = self.enc2(xc, yc)
        encoded = torch.cat([theta1,
            torch.stack([theta2]*xt.shape[-2], -2)], -1)
        return encoded

    def predict(self, xc, yc, xt, num_samples=None, return_base=False):
        with torch.no_grad():
            bxc, byc = SWR(xc, yc, num_samples=num_samples)
            sxc, syc = stack(xc, num_samples), stack(yc, num_samples)

            encoded = self.encode(bxc, byc, sxc)
            py_res = self.dec(encoded, sxc)

            mu, sigma = py_res.mean, py_res.scale
            res = SWR((syc - mu)/sigma).detach()
            res = (res - res.mean(-2, keepdim=True))

            bxc = sxc
            byc = mu + sigma * res

        encoded_base = self.encode(xc, yc, xt)

        sxt = stack(xt, num_samples)
        encoded_bs = self.encode(bxc, byc, sxt)

        py = self.dec(stack(encoded_base, num_samples),
                sxt, ctx=encoded_bs)

        # if self.training or return_base:
        #     py_base = self.dec(encoded_base, xt)
        #     return py_base, py
        # else:
        #     return py
        return py.mean

    def forward(self, xc, yc, x, y, num_samples=2, reduce_ll=True):
        # outs = AttrDict()

        # def compute_ll(py, y):
        #     ll = py.log_prob(y).sum(-1)
        #     if ll.dim() == 3 and reduce_ll:
        #         ll = logmeanexp(ll)
        #     return ll

        py = self.predict(xc, yc, x, num_samples=num_samples)

        return py, y



class BootstrappingAttentiveNeuralProcess(nn.Module):
    """
    Latent Model (Attentive Neural Process)
    """
    def __init__(
        self,
        cfg_dim_input=None,
        cfg_dim_output=None,
        d_model=None,
        # d_model=None,          # fuck!!!!
        d_inner=None,
        n_layers=None,
        # n_head=None,
        n_head=None,
        d_k=None,
        d_v=None,
    ):
        super(BootstrappingAttentiveNeuralProcess, self).__init__()

        self.banp = BANP(
            dim_x=cfg_dim_input-1,
            dim_y=1,
            dim_hid=d_model,
            enc_v_depth=4,
            enc_qk_depth=2,
            enc_pre_depth=4,
            enc_post_depth=2,
            dec_depth=3
        )


    def forward(self, context_x, context_y, target_x, target_y=None):

        y_pred, target_y = self.banp(context_x, context_y, target_x, target_y)

        y_pred = y_pred.mean(dim=0)

        return y_pred, target_y

    
    # def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
    #     kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
    #     kl_div = 0.5 * kl_div.sum()
    #     return kl_div







# class AttentiveNeuralProcess(nn.Module):
#     def __init__(
#         self,
#         input_dim=None,
#         output_dim=None,

#         cfg_dim_input=None,
#         d_model=None,
#         # d_model=None,          # fuck!!!!
#         d_inner=None,
#         n_layers=None,
#         # n_head=None,
#         n_head=None,
#         d_k=None,
#         d_v=None,
#     ):
#         super(AttentiveNeuralProcess, self).__init__()

#         encoder_sizes=[cfg_dim_input, 128, 128, 128, 256]
#         decoder_sizes=[256 + cfg_dim_input-1, 256, 256, 128, 128, 2]

#         self._encoder = DeterministicEncoder(encoder_sizes)
#         self._decoder = DeterministicDecoder(decoder_sizes)

#     def forward(self, x_context, y_context, x_target, y_target=None):
#         representation = self._encoder(x_context, y_context)
#         dist, mu, sigma = self._decoder(representation, x_target)

#         # log_p = None if y_target is None else dist.log_prob(y_target)
#         # return log_p, mu, sigma
#         return mu, y_target
