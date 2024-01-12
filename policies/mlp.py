#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mlp.py
@Time    :   2023/04/21 14:55:39
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   from PAERL
'''


import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from einops import repeat, rearrange
def identity(x):
    return x

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output


class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        #self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output
        
class MlpEncoder(Mlp):
    def reset(self, num_tasks=1):
        pass


# class RecurrentEncoder(Mlp):
#     '''
#     encode context via recurrent network
#     '''

#     def __init__(self,
#                  device = 'cpu',
#                  *args,
#                  **kwargs
#     ):
#         #self.save_init_params(locals())
#         super().__init__(*args, **kwargs)
#         self.hidden_dim = self.hidden_sizes[-1]
#         self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))
#         self.device = device
#         # input should be (task, seq, feat) and hidden should be (task, 1, feat)

#         self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

#     def forward(self, in_, return_preactivations=False,return_last=True):
#         # expects inputs of dimension (task, seq, feat)
#         task, seq, feat = in_.size()
#         out = rearrange(in_, 'b s f -> (b s) f')
#         #out = in_.view(task * seq, feat)

#         # embed with MLP
#         for i, fc in enumerate(self.fcs):
#             out = fc(out)
#             out = self.hidden_activation(out)

#         out = rearrange(out, '(b s) f -> b s f', b=task, s=seq)
#         #out = out.view(task, seq, -1)
#         out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(self.device)))
#         self.hidden = hn
#         # take the last hidden state to predict z
#         if return_last:
#             out = out[:, -1, :]

#         # output layer
#         preactivation = self.last_fc(out)
#         output = self.output_activation(preactivation)
#         if return_preactivations:
#             return output, preactivation
#         else:
#             return output

#     def reset(self, num_tasks=1):
#         self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)


class RecurrentEncoder(Mlp):
    '''
    encode context via recurrent network
    I rewrite it. initial hidden for traj
    '''

    def __init__(self,
                 device = 'cpu',
                 *args,
                 **kwargs
    ):
        #self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.device = device
        # self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))
        # input should be (seq, batch, feat) 

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=False) ## only one layer

    def forward(self, in_, return_preactivations=False,return_last=True):
        # expects inputs of dimension (task, seq, feat)
        seq, batch, feat = in_.size()
        out = rearrange(in_, 's b f -> (s b) f')
        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)
        out = rearrange(out, '(s b) f -> s b f', b=batch, s=seq)

        self.init_hidden_state(batch_size=batch)
        out, (hn, cn) = self.lstm(out, (self.hidden, self.cells))
        self.hidden = hn
        # take the last hidden state to predict z
        if return_last:
            out = out[-1,: , :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output
        
    def init_hidden_state(self, batch_size=1):
        self.hidden = torch.zeros(1, batch_size, self.hidden_dim) 
        self.cells = torch.zeros(1, batch_size, self.hidden_dim)

    # def reset(self, num_tasks=1):
    #     self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0) 





