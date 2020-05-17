# -*- coding: utf-8 -*-
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable

def activation(input):
    return F.selu(input)

def MSEloss(inputs, targets, size_average=False):
    mask = targets != 0
    num_ratings = torch.sum(mask.float())
    criterion = nn.MSELoss(reduction='sum' if not size_average else 'mean')
    return criterion(inputs * mask.float(), targets), Variable(torch.Tensor([1.0])) if size_average else num_ratings

class AutoEncoder(nn.Module):
    def __init__(self, layer_sizes=[737149, 512, 512, 1024], dp_drop_prob=0.8, last_layer_activations=True):
        super(AutoEncoder, self).__init__()
        self._dp_drop_prob = dp_drop_prob
        if dp_drop_prob > 0:
            self.drop = nn.Dropout(dp_drop_prob)
        self._last_layer_activations = last_layer_activations
        self._last = len(layer_sizes) - 2
        
        self.encode_w = nn.ParameterList(
            [nn.Parameter(torch.rand(layer_sizes[i+1], layer_sizes[i])) for i in range(len(layer_sizes) - 1)])

        for idx, w in enumerate(self.encode_w):
            weight_init.xavier_uniform_(w)

        self.encode_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)])

        reversed_enc_layers = list(reversed(layer_sizes))

        self.decode_w = nn.ParameterList(
            [nn.Parameter(torch.rand(reversed_enc_layers[i+1], reversed_enc_layers[i])) for i in range(len(reversed_enc_layers) -1)])
        for idx, w in enumerate(self.decode_w):
            weight_init.xavier_uniform_(w)

        self.decode_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(reversed_enc_layers[i+1])) for i in range(len(reversed_enc_layers) - 1)])

    def encoder(self, x):
        for idx, w in enumerate(self.encode_w):
            x = activation(input = F.linear(input=x, weight=w, bias=self.encode_b[idx]))
        if self._dp_drop_prob > 0:
            x = self.drop(x)
        return x

    def decoder(self, z):
        for idx, w in enumerate(list(reversed(self.encode_w))):
            if idx != self._last or self._last_layer_activations:
                z = activation(input=F.linear(input=z, weight=w.transpose(0, 1), bias=self.decode_b[idx]))
            else:
                z = F.linear(input=z, weight=w.transpose(0, 1), bias=self.decode_b[idx])
        return z

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
