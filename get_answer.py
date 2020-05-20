# -*- coding:utf-8 -*-
import os
import sys

from time import perf_counter as pc
import pickle as pkl
from tqdm import tqdm

import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader

from arena_util import *
from autoencoder import AutoEncoder

class GetAnswer:
    
    def __init__(self, meta_fname, model_fname, question_fname, result_fname):
                
        with open(meta_fname, 'rb') as fp:
            meta = pkl.load(fp)
        self.song_length = meta['song_length']
        self.tag_length = meta['tag_length']
        self.tag2idx = meta['tag2idx']
        self.idx2tag = {idx:tag for tag, idx in self.tag2idx.items()}
        self.model_fname = model_fname
        self.question_fname = question_fname
        self.result_fname = result_fname

    def _make_array(self, data):
        songs = torch.zeros(self.song_length)
        tags = torch.zeros(self.tag_length)
        songs[data['songs']] = 1.
        tag_idx = [self.tag2idx[tag] for tag in data['tags']]
        tags[tag_idx] = 1.
        return torch.cat((songs, tags))

    def _generate_answers(self, questions):
        model = AutoEncoder()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device : {}'.format(device))
        
        model.to(device)
        print("Load model's parameter ... ")
        model.load_state_dict(torch.load(self.model_fname))
        
        model.eval()
        
        answers = []
        
        for question in tqdm(questions):
            inputs = self._make_array(question).to(device)
            outputs = model(inputs)
            res = outputs.cpu().detach().numpy()
            song_rating = res[:self.song_length]
            tag_rating = res[self.song_length:]
            # songs = sorted(range(len(song_rating)), key=lambda x:song_rating[x], reverse=True)[:200]
            # tag_idx = sorted(range(len(tag_rating)), key=lambda x:tag_rating[x], reverse=True)[:100]
            songs = np.argsort(song_rating)[::-1][:200]
            tag_idx = np.argsort(tag_rating)[::-1][:100]
            tags = [ self.idx2tag[idx] for idx in tag_idx]
            answers.append({
                "id" : question["id"],
                "songs" : remove_seen(question["songs"], songs)[:100],
                "tags" : remove_seen(question["tags"], tags)[:10],
            })
        return answers
        
    def run(self):
        print('Loading question file ... ')
        questions = load_json(self.question_fname)

        print("Writing answers ... ")
        answers = self._generate_answers(questions)
        write_json(answers, self.result_fname)

if __name__ == '__main__':
    get_answer = GetAnswer(meta_fname = './data/meta.pkl',
                           model_fname = './res/model/deepreco_11',
                           question_fname = './arena_data/questions/val.json',
                           result_fname = './results/test.json')
    get_answer.run()
