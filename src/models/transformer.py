""" Define the Transformer model """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ["TransformerRecon"]


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


# Modified DecoderLayer
class DecoderLayerModify(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayerModify, self).__init__()
        # self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_query, enc_key, enc_output):
        dec_output, dec_enc_attn = self.enc_attn(dec_query, enc_key, enc_output)

        self.fuck_attn = dec_enc_attn

        dec_output = self.pos_ffn(dec_output)
        return dec_output



# Modified Encoder
class EncoderModify(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super(EncoderModify, self).__init__()
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq):
        enc_output = src_seq
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)

        return enc_output


# Modified Decoder
class DecoderModify(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super(DecoderModify, self).__init__()
        self.layer_stack = nn.ModuleList(
            [
                DecoderLayerModify(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, trg_seq, src_key, enc_output):
        dec_output = trg_seq
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, src_key, enc_output)
        return dec_output


# Modified Transformer
class TransformerRecon(nn.Module):
    def __init__(
        self,
        cfg_dim_input=3,
        cfg_dim_output=None,
        d_model=128,
        d_inner=512,
        n_layers=3,
        n_head=4,
        # d_k=32,
        # d_v=32,
        dropout=0.0,
    ):

        d_k = d_model // n_head
        d_v = d_model // n_head

        super(TransformerRecon, self).__init__()

        # Embedding
        self.emb_src = nn.Sequential(
            nn.Linear(cfg_dim_input, d_model),
            # nn.GELU(),
            # nn.Linear(256, 256),
        )
        self.emb_trg = nn.Sequential(
            nn.Linear(cfg_dim_input-1, d_model),
            # nn.GELU(),
            # nn.Linear(256, 256),
        )

        self.d_model = d_model

        self.encoder = EncoderModify(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.decoder = DecoderModify(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=1,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.pre = nn.Sequential(
            nn.Linear(d_model + cfg_dim_input-1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, src_seq, src_label, trg_seq, trg_label=None):

        src_rep = torch.cat([src_seq, src_label], dim=2)
        trg_x = trg_seq
        src_rep = self.emb_src(src_rep)
        trg_seq = self.emb_trg(trg_seq)
        src_seq = self.emb_trg(src_seq)
        enc_output = self.encoder(src_rep)

        dec_output = self.decoder(trg_seq, src_seq, enc_output)
        return self.pre(torch.cat([dec_output, trg_x], dim=-1)), trg_label

