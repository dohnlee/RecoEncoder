# -*- coding: utf-8 -*-
import os
import sys

import time

from math import sqrt

import numpy as np
import pickle as pkl

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.model_selection import train_test_split

from arena_util import load_json
from autoencoder import *
 
class PlayListDataset(Dataset):
    def __init__(self, dataset, meta_fname):
        self.data = dataset # numpy array
        with open(meta_fname, 'rb') as fp:
            meta = pkl.load(fp)
        self.song_length = meta['song_length']
        self.tag_length = meta['tag_length']
        self.tag2idx = meta['tag2idx']
        self.len = len(self.data)
    def _make_array(self, data):
        songs = torch.zeros(self.song_length)
        tags = torch.zeros(self.tag_length)
        songs[data['songs']] = 1.
        tag_idx = [self.tag2idx[tag] for tag in data['tags']]
        tags[tag_idx] = 1.
        return torch.cat((songs, tags))
    def __getitem__(self, index):
        # tmp = self.data[index]
        return self._make_array(self.data[index])
    def __len__(self):
        return self.len

def main():
    data_fname = './data/train.json'
    meta_fname = './data/meta.pkl'
    result_fname = './res/model/deepreco'
    batch_size = 32
    drop_prob = 0.8
    lr = 0.005
    num_epochs = 101
    noise_prob = 0
    aug_step = 1

    # train-val split
    raw_data = load_json(data_fname)
    train, val = train_test_split(np.array(raw_data), train_size=0.8, random_state=128)
    train_dataset = PlayListDataset(train, meta_fname)
    val_dataset = PlayListDataset(val, meta_fname)
    
    # data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)
    
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)

    # check the model
    model = AutoEncoder()
    print('model : ')
    print(model)

    # check available gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device : {}'.format(device))
   
    model.to(device)

    # set optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0)
    scheduler = MultiStepLR(optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5)
    
    # de-noise
    if noise_prob > 0:
        dp = nn.Dropout(p=noise_prob)
    
    # train
    train_loss_array = []
    val_loss_array = []
    for epoch in range(num_epochs):
        t_denom = 0.0
        total_t_loss = 0.0
        print('Doing epoch {} / {}'.format(epoch+1, num_epochs))
        model.train()
        # lr scheduling
        scheduler.step()
        start = time.perf_counter()
        for pl in tqdm(train_loader):
            pl = pl.view(pl.size(0), -1)
            inputs = Variable(pl.to(device))
            # forward
            if noise_prob > 0:
                inputs_noise = dp(inputs) * (1 - noise_prob)
                outputs = model(inputs_noise)
            else:
                outputs = model(inputs)
            loss, num_song_tags = MSEloss(outputs, inputs)
            # backward
            optimizer.zero_grad()
            loss = loss/num_song_tags
            loss.backward()
            optimizer.step()
            t_denom += 1.
            total_t_loss += loss.item()

            if aug_step > 0:
                for t in range(aug_step):
                    inputs = Variable(outputs.data)
                    if noise_prob > 0.0:
                        inputs = dp(inputs)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss, num_song_tags = MSEloss(outputs, inputs)
                    loss = loss / num_song_tags
                    loss.backward()
                    optimizer.step()

        elapsed = time.perf_counter() - start
        t_loss = sqrt(total_t_loss / t_denom)
        print('train loss : {:.4f}'.format(t_loss))
        print('elapsed : {:.4f} sec'.format(elapsed))
        train_loss_array.append(t_loss)
        
        # print('Doing validation ...')
        model.eval()
        denom = 0.0
        total_epoch_loss = 0.0
        for pl in val_loader:
            pl = pl.view(pl.size(0), -1)
            inputs = Variable(pl.to(device))
            outputs = model(inputs)
            loss, num_song_tags = MSEloss(outputs, inputs)
            total_epoch_loss += loss.item()
            denom += num_song_tags.item() 
        val_loss = sqrt(total_epoch_loss / denom)
        print('val loss : {:.4f}'.format(val_loss))
        val_loss_array.append(val_loss)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), '{}_{}'.format(result_fname, epoch+1)) 
            with open('{}_loss'.format(result_fname), 'wb') as fp:
                pkl.dump((train_loss_array, val_loss_array), fp)

if __name__ == '__main__':
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)
    torch.manual_seed(128)
    main()
