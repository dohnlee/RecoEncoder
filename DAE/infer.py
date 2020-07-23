# -*- coding:utf-8 -*-
import os
import sys

import json
import pickle as pkl
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import xgboost

from util import *
from DAE.data import PlayListDataset
from DAE.network import *

class Inference(PlayListDataset):
    def __init__(self, meta, device):
        
        self.song2idx = meta['song2idx']
        self.tag2idx = meta['tag2idx']
        self.token2idx = meta['token2idx']
        self.song_length = len(self.song2idx)
        self.tag_length = len(self.tag2idx)
        self.token_length = len(self.token2idx)
        self.idx2song = {idx:song for song, idx in self.song2idx.items()}
        self.idx2tag = {idx:tag for tag, idx in self.tag2idx.items()}
        if device == 'gpu':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def inference(self, model, questions, num_songs, num_tags, with_score=False):
        model.eval()
        results = []
        if with_score:
            scores = []
        with torch.no_grad():
            for question in tqdm(questions):

                tokens, songs, tags = self._make_array(question)

                inputs = torch.cat((tokens, songs, tags))
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                res = outputs.cpu().detach().numpy()

                song_rating = res[self.token_length:self.token_length+self.song_length]
                tag_rating = res[self.token_length+self.song_length:]
                
                song_idx = np.argsort(song_rating)[::-1][:num_songs+100]
                tag_idx = np.argsort(tag_rating)[::-1][:num_tags+90]
                
                if with_score:
                    song2score = {self.idx2song[idx]:song_rating[idx] for idx in song_idx}
#                    tag2score = {self.idx2tag[idx]:tag_rating[idx] for idx in tag_idx}
                    scores.append({
                            "id":question['id'],
                            "song2score":song2score
#                            "tag2score":tag2score
                    })
                songs = [self.idx2song[idx] for idx in song_idx]
                tags = [self.idx2tag[idx] for idx in tag_idx]
                
                results.append({
                    "id":question['id'],
                    "songs":remove_seen(question['songs'], songs)[:num_songs],
                    "tags":remove_seen(question['tags'], tags)[:num_tags]
                })
        if with_score:
            return results, scores
        else:
            return results

    def ranking(self, args, questions, candidates, scores, song_df, co_song):
        model = load_pickle(args.xg_fname)
        results = []
        c_dict = {c['id']:c for c in candidates}
        s_dict = {score['id']:score for score in scores}
        for question in tqdm(questions):
            _id = question['id']
            seeds = question['songs']
            if len(seeds) == 0:
                results.append({
                    "id":_id,
                    "songs" : c_dict[_id]['songs'][:100],
                    "tags" : c_dict[_id]['tags']
                })
            else:
                song2score = s_dict[_id]['song2score']
                candis = c_dict[_id]['songs']
                c2r = dict()
                inputs = []
                for candi in candis:
                    rating = song2score[candi]
                    tmp, tmp_n = [], []
                    for seed in seeds:
                        if seed in co_song[candi]:
                            co = co_song[candi][seed]
                            tmp.append(co)
                            tmp_n.append(co/song_df[seed])
                        else:
                            tmp.append(0)
                            tmp_n.append(0)
                    tmp, tmp_n = np.array(tmp), np.array(tmp_n)
                    _input = [tmp.min(), tmp.max(), tmp.mean(),
                            tmp_n.min(), tmp_n.max(), tmp_n.mean(), rating]
                    inputs.append(_input)
                inputs = xgboost.DMatrix(np.array(inputs))
                r_score = model.predict(inputs)
                c2r = list(zip(candis, r_score))
                songs = sorted(c2r, key=lambda x:x[1], reverse=True)[:100]
                results.append({
                    "id":_id,
                    "songs":[song[0] for song in songs],
                    "tags":c_dict[_id]['tags']
                })
        return results
                
