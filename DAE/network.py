# -*- coding: utf-8 -*-
import os
import sys

import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable

def BCEloss(inputs, targets, reduction='mean'):
    mask = targets == 0
    weights = 0.55 * mask.float()
    mask = mask == 0
    weights += mask.float()
    loss = nn.BCELoss(weight=weights, reduction=reduction)
    return loss(inputs, targets)

class AutoEncoder(nn.Module):
    def __init__(self, input_size=232907, layer_sizes=[512, 512, 1024], dp_drop_prob=0.0):
        super(AutoEncoder, self).__init__()
        self._dp_drop_prob = dp_drop_prob
        if dp_drop_prob > 0:
            self.drop = nn.Dropout(dp_drop_prob)
        self._last = len(layer_sizes) - 1
        
        layer_sizes.insert(0, input_size)
        self.encode_w = nn.ParameterList(
            [nn.Parameter(torch.rand(layer_sizes[i+1], layer_sizes[i])) for i in range(len(layer_sizes) -1)])
        
        for w in self.encode_w:
            weight_init.xavier_uniform_(w)
        
        self.encode_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)])

        reversed_enc_layers = list(reversed(layer_sizes))
        
        self.decode_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(reversed_enc_layers[i+1])) for i in range(len(reversed_enc_layers) - 1)])

    def encoder(self, x):
        for idx, w in enumerate(self.encode_w):
            x = torch.selu(input=F.linear(input=x, weight=w, bias=self.encode_b[idx]))
        if self._dp_drop_prob > 0:
            x = self.drop(x)
        return x

    def decoder(self, z):
        for idx, w in enumerate(list(reversed(self.encode_w))):
            if idx != self._last:
                z = torch.selu(input=F.linear(input=z, weight=w.transpose(0, 1), bias=self.decode_b[idx]))
            else:
                z = torch.sigmoid(F.linear(input=z, weight=w.transpose(0, 1), bias=self.decode_b[idx]))
        return z

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

