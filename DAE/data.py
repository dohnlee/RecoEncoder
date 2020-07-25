# -*- coding: utf-8 -*-
import os
import sys
import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from khaiii import KhaiiiApi

from util import *

class PlayListDataset(Dataset):
    def __init__(self, dataset, song2idx, tag2idx, token2idx):
        self.data = dataset 
        self.song2idx = song2idx
        self.tag2idx = tag2idx
        self.token2idx = token2idx
        self.song_length = len(song2idx)
        self.tag_length = len(tag2idx)
        self.token_length = len(token2idx)
        self.len = len(self.data)
            
    def _make_array(self, data):
        tokens = torch.zeros(self.token_length)
        songs = torch.zeros(self.song_length)
        tags = torch.zeros(self.tag_length)
        token_idx = [self.token2idx[token] for token in data['tokens'] if token in self.token2idx]
        tokens[token_idx] = 1.
        song_idx = [self.song2idx[song] for song in data['songs'] if song in self.song2idx]
        songs[song_idx] = 1.
        tag_idx = [self.tag2idx[tag] for tag in data['tags'] if tag in self.tag2idx]
        tags[tag_idx] = 1.
        return tokens, songs, tags

    def __getitem__(self, index):
        return self._make_array(self.data[index])
    
    def __len__(self):
        return self.len

class PreProcess(object):
    def __init__(self):
        self.api = KhaiiiApi()

    def tokenizer(self, text, pos_set={'NNG', 'NNP', 'NNB', 'NP', 'VV', 'VA', 'SL'}):
        tokens = []
        data = self.api.analyze(text)
        for word in data:
            tokens.extend([str(m).split('/')[0].lower() for m in word.morphs
                if str(m).split('/')[1] in pos_set])
        return tokens

    def make_meta(self, data):
        song_df = defaultdict(int)
        tag_df = defaultdict(int)
        token_df = defaultdict(int)

        print('make meta ... ')
        for i in tqdm(range(len(data))):
            songs = data[i]['songs']
            tags = data[i]['tags']
            title = data[i]['plylst_title']

            for song in songs:
                song_df[song] += 1

            for tag in tags:
                tag_df[tag] += 1

            try:
                tokens = [token for token in self.tokenizer(title)]
                for token in set(tokens):
                    token_df[token] += 1
                data[i]["tokens"] = tokens
            except:
                data[i]["tokens"] = []

        song_over_3_ids = sorted([_ for _, df in song_df.items() if df >=3])
        tag_over_5_ids = sorted([_ for _, df in tag_df.items() if df >=5])
        token_over_5_ids = sorted([_ for _, df in token_df.items() if df >=5])
        
        song_length = len(song_over_3_ids)
        tag_length = len(tag_over_5_ids)
        token_length = len(token_over_5_ids)

        song2idx = {song_over_3_ids[i]:i for i in range(song_length)}
        tag2idx = {tag_over_5_ids[i]:i for i in range(tag_length)}
        token2idx = {token_over_5_ids[i]:i for i in range(token_length)}
        
        print('meta file info')
        print('song length : ', song_length)
        print('tag length : ', tag_length)
        print('token length : ', token_length)
        
        meta = {"song2idx" : song2idx,
                "tag2idx" : tag2idx,
                "token2idx" : token2idx,
                "song_df" : dict(song_df),
                "tag_df" : dict(tag_df)}

        return meta

    def make_input(self, data, meta):
        token2idx = meta['token2idx']
        for i in tqdm(range(len(data))):
            title = data[i]['plylst_title']
            try:
                tokens = [token for token in self.tokenizer(title) if token in token2idx]
                data[i]["tokens"] = tokens
            except:
                data[i]["tokens"] = []

        return data
    
    def make_codict(self, trains, questions, meta):
        print('make codict ... ')
        song2idx = meta['song2idx']
        song_df = defaultdict(int)
        co_song = {song:defaultdict(int) for song in song2idx}
        
        for data in [trains, questions]:
            for i in range(len(data)):
                songs = [song for song in data[i]['songs'] if song in song2idx]
                for song in songs:
                    song_df[song]+=1

            for i in tqdm(range(len(data))):
                songs = [song for song in data[i]['songs'] if song in song2idx]
                for j in range(len(songs)):
                    seed = songs[j]
                    for song in songs[j:]:
                        if song != seed:
                            co_song[song][seed] += 1
                            co_song[seed][song] += 1
            
        return dict(co_song), song_df
    
    def make_xgboost_input(self, args, questions, candidates, scores, co_song, song_df, answers=None):
        if not (len(questions) == len(candidates) == len(scores)):
                sys.exit('Error: seeds, candidates, socres, answers are not coincide')
        
        cnt = 0
        check_point = 1
        inputs=[]
        c_dict = {c['id']:c for c in candidates}
        s_dict = {s['id']:s for s in scores}
        if answers is not None:
            a_dict = {a['id']:a for a in answers}
        for question in tqdm(questions):
            _id = question['id']
            seeds = question['songs']
            seeds = [seed for seed in seeds if seed in co_song]
            if len(seeds) == 0:
                continue
            song2score = s_dict[_id]['song2score']
            candis = c_dict[_id]['songs']
            if answers is not None:
                a_songs = set(a_dict[_id]['songs'])
            
            for candi in candis:
                rating=song2score[candi]
                if answers is not None:
                    if candi in a_songs:
                        label = 1
                    else:
                        label = 0
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
                if answers is not None:
                    _input.append(label)
                inputs.append(_input)
                cnt += 1
                if cnt % 10000000 == 0:
                    inputs = np.array(inputs)
                    if not os.path.isdir(args.xg_input_fname):
                        os.makedirs(args.xg_input_fname)
                    write_pickle(inputs, os.path.join(args.xg_input_fname, 'train_{}'.format(check_point)))
                    cnt = 0
                    check_point += 1
                    inputs = []
        inputs = np.array(inputs)
        write_pickle(inputs, os.path.join(args.xg_input_fname, 'train_{}'.format(check_point)))

class ArenaSplitter(object):
    def split_data(self, playlists, ratio=0.8):
        tot = len(playlists)
        train = playlists[:int(tot*ratio)]
        val = playlists[int(tot*ratio):]

        return train, val

    def _mask(self, playlists, mask_cols, del_cols):
        q_pl = copy.deepcopy(playlists)
        a_pl = copy.deepcopy(playlists)

        for i in range(len(playlists)):
            for del_col in del_cols:
                q_pl[i][del_col] = []
                if del_col == 'songs':
                    a_pl[i][del_col] = a_pl[i][del_col][:100]
                elif del_col == 'tags':
                    a_pl[i][del_col] = a_pl[i][del_col][:10]

            for col in mask_cols:
                mask_len = len(playlists[i][col])
                mask = np.full(mask_len, False)
                mask[:mask_len//2] = True
                np.random.shuffle(mask)

                q_pl[i][col] = list(np.array(q_pl[i][col])[mask])
                a_pl[i][col] = list(np.array(a_pl[i][col])[np.invert(mask)])

        return q_pl, a_pl

    def mask_data(self, playlists):
        playlists = copy.deepcopy(playlists)
        tot = len(playlists)
        song_only = playlists[:int(tot * 0.3)]
        song_and_tags = playlists[int(tot * 0.3):int(tot * 0.8)]
        tags_only = playlists[int(tot * 0.8):int(tot * 0.95)]
        title_only = playlists[int(tot * 0.95):]

        print(f"Total: {len(playlists)}, "
              f"Song only: {len(song_only)}, "
              f"Song & Tags: {len(song_and_tags)}, "
              f"Tags only: {len(tags_only)}, "
              f"Title only: {len(title_only)}")

        song_q, song_a = self._mask(song_only, ['songs'], ['tags'])
        songtag_q, songtag_a = self._mask(song_and_tags, ['songs', 'tags'], [])
        tag_q, tag_a = self._mask(tags_only, ['tags'], ['songs'])
        title_q, title_a = self._mask(title_only, [], ['songs', 'tags'])

        q = song_q + songtag_q + tag_q + title_q
        a = song_a + songtag_a + tag_a + title_a

        shuffle_indices = np.arange(len(q))
        np.random.shuffle(shuffle_indices)

        q = list(np.array(q)[shuffle_indices])
        a = list(np.array(a)[shuffle_indices])

        return q, a

