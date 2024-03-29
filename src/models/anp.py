import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


# code from https://github.com/soobinseo/Attentive-Neural-Process

__all__ = ["AttentiveNeuralProcess"]


class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)

class LatentEncoder(nn.Module):
    """
    Latent Encoder [For prior, posterior]
    """
    def __init__(self, num_hidden, num_latent, input_dim=3):
        super(LatentEncoder, self).__init__()
        self.input_projection = Linear(input_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

    def forward(self, x, y):
        # concat location (x) and value (y)
        encoder_input = torch.cat([x,y], dim=-1)
        
        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)
        
        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)
        
        # mean
        hidden = encoder_input.mean(dim=1)
        hidden = F.gelu(self.penultimate_layer(hidden))
        
        # get mu and sigma
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)
        
        # reparameterization trick
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        
        # return distribution
        return mu, log_sigma, z
    
class DeterministicEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """
    def __init__(self, num_hidden, num_latent, input_dim):
        super(DeterministicEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(input_dim, num_hidden)
        self.context_projection = Linear(input_dim-1, num_hidden)
        self.target_projection = Linear(input_dim-1, num_hidden)

    def forward(self, context_x, context_y, target_x):
        # concat context location (x), context value (y)
        encoder_input = torch.cat([context_x,context_y], dim=-1)
        
        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)
        
        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)
        
        # query: target_x, key: context_x, value: representation

        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)
        
        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)
        
        return query
    
class Decoder(nn.Module):
    """
    Decoder for generation 
    """
    def __init__(self, num_hidden, input_dim):
        super(Decoder, self).__init__()
        self.target_projection = Linear(input_dim, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 3, num_hidden * 3, w_init='relu') for _ in range(3)])
        self.final_projection = Linear(num_hidden * 3, 1)
        
    def forward(self, r, z, target_x):
        batch_size, num_targets, _ = target_x.size()
        # project vector with dimension 2 --> num_hidden
        target_x = self.target_projection(target_x)
        
        # concat all vectors (r,z,target_x)
        hidden = torch.cat([torch.cat([r,z], dim=-1), target_x], dim=-1)
        
        # mlp layers
        for linear in self.linears:
            hidden = F.gelu(linear(hidden))
            
        # get mu and sigma
        y_pred = self.final_projection(hidden)
        
        return y_pred

class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """
    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden 
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        # Get attention score
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)
        
        attn = torch.softmax(attn, dim=-1)

        # Dropout
        attn = self.attn_dropout(attn)
        
        # Get Context Vector
        result = torch.bmm(attn, value)

        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """
    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads 
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query):

        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query
        
        # Make multihead
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)
        
        # Concatenate context vector with input (most important)
        result = torch.cat([residual, result], dim=-1)
        
        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns


class AttentiveNeuralProcess(nn.Module):
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

        super(AttentiveNeuralProcess, self).__init__()

        num_hidden = d_model

        self.latent_encoder = LatentEncoder(num_hidden, num_hidden, input_dim=cfg_dim_input)
        self.deterministic_encoder = DeterministicEncoder(num_hidden, num_hidden, input_dim=cfg_dim_input)
        self.decoder = Decoder(num_hidden, input_dim=cfg_dim_input-1)
        self.BCELoss = nn.BCELoss()


    def forward(self, context_x, context_y, target_x, target_y=None):
        num_targets = target_x.size(1)
        
        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)
        
        # For training
        # if target_y is not None:
        #     posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)
        #     z = posterior
        
        # # For Generation
        # else:
        #     z = prior

        z = prior
        
        z = z.unsqueeze(1).repeat(1,num_targets,1) # [B, T_target, H]
        r = self.deterministic_encoder(context_x, context_y, target_x) # [B, T_target, H]
        
        # mu should be the prediction of target y
        y_pred = self.decoder(r, z, target_x)
        
        # # For Training
        # if target_y is not None:
        #     # get log probability
        #     bce_loss = self.BCELoss(t.sigmoid(y_pred), target_y)
            
        #     # get KL divergence between prior and posterior
        #     kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var)
            
        #     # maximize prob and minimize KL divergence
        #     loss = bce_loss + kl
        
        # # For Generation
        # else:
        #     log_p = None
        #     kl = None
        #     loss = None
        
        # return y_pred, kl, loss
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
