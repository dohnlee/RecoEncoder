# -*- coding: utf-8 -*-
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from util import *
from DAE.data import PlayListDataset
from DAE.network import *
from DAE.infer import Inference

class TrainAE(PlayListDataset):
    def __init__(self, meta, device):
        self.meta = meta
        self.song2idx = meta['song2idx']
        self.tag2idx = meta['tag2idx']
        self.token2idx = meta['token2idx']
        self.song_length = len(self.song2idx)
        self.tag_length = len(self.tag2idx)
        self.token_length = len(self.token2idx)
        self.idx2song = {idx:song for song, idx in self.song2idx.items()}
        self.idx2tag = {idx:tag for tag, idx in self.tag2idx.items()}
        self.infer = Inference(meta, device)
        if device == 'gpu':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def main(self, args, results_fname, trains, vals=None):
        # data
        random.seed(128)
        # trains = random.sample(trains, 1000)
        if vals is not None:
            questions, answers = vals
            ndcg = Ndcg(answers)

        train_dataset = PlayListDataset(dataset=trains,
                song2idx=self.song2idx,
                tag2idx=self.tag2idx,
                token2idx=self.token2idx)
        
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0)
        
        input_size = self.token_length + self.song_length + self.tag_length

        # check the model
        model = AutoEncoder(input_size=input_size, dp_drop_prob=args.drop_prob)

        print("model's parameter size: ")
        for param in model.parameters():
            print(param.size())
        
        print('device : {}'.format(self.device))
        model.to(self.device)
        
        # set optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        
        # de-noise
        if args.noise_prob > 0:
            dp = nn.Dropout(p=args.noise_prob)
        
        # train
        np.random.seed(128)
        train_loss_array = []
        ndcg_array = []
        check_score = None
        for epoch in range(1, args.epochs + 1):
            t_denom = 0.0
            total_t_loss = 0.0
            print('Doing epoch {} / {}'.format(epoch, args.epochs))
            model.train()
            for tokens, songs, tags in tqdm(train_loader):
                inputs = torch.cat((tokens ,songs, tags), dim=1)
                inputs = Variable(inputs.to(self.device))
                
                input_noise = inputs.clone()
                # hide and seek
                r = np.random.randint(4)
                if r == 0:
                    input_noise[:, self.token_length:] = 0.
                elif r == 1:
                    input_noise[:, :self.token_length] = 0.
                elif r == 2:
                    input_noise[:, :self.token_length] = 0.
                    input_noise[:, self.token_length + self.song_length:] = 0.
                else:
                    input_noise[:, :self.token_length + self.song_length] = 0.

                # forward(+denoising)
                if args.noise_prob > 0:
                    input_noise = dp(input_noise) * (1 - args.noise_prob)
                    outputs = model(input_noise)
                else:
                    outputs = model(input_noise)
                
                loss = BCEloss(outputs, inputs)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t_denom += 1.
                total_t_loss += loss.item()
                
            t_loss = total_t_loss/t_denom
            print('train loss : {:.4f}'.format(t_loss))
            train_loss_array.append(t_loss)
            
            # lrdecay 
            scheduler.step()
            
            # validation
            if vals is not None:
                if epoch % 5 == 0:
                    print('calculate NDCG')
                    results = self.infer.inference(model, questions, num_songs=100, num_tags=10)
                    eval_ndcg = ndcg.main(results)
                    ndcg_array.append(eval_ndcg)
                    print('music ndcg : {:.4f}, tag ndcg : {:.4f}, score : {:.4f}'.format(*eval_ndcg))
                    if check_score is None:
                        check_score = eval_ndcg[-1]
                        torch.save(model.state_dict(), results_fname)
                    elif check_score < eval_ndcg[-1]:
                        check_score = eval_ndcg[-1]
                        torch.save(model.state_dict(), results_fname)

            if epoch % args.check_point == 0:
                torch.save(model.state_dict(), '{}_dict_{}'.format(results_fname, epoch)) 
                if vals is not None:
                    write_pickle((train_loss_array, ndcg_array), '{}_loss'.format(results_fname))
            
        torch.save(model.state_dict(), results_fname)
        return model

