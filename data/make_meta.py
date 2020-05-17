# -*- coding: utf-8 -*-
import fire
import pickle as pkl
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict

def get_meta(song_fname, train_fname, result_fname):
    
    print('Loading data ... ')
    song_meta = pd.read_json(song_fname)
    song_length = song_meta.shape[0]
    train = pd.read_json(train_fname)
    
    print('Get tag info')
    tag_length = 0
    tag2idx = defaultdict()
    for i in tqdm(range(len(train))):
        tags = train.loc[i, 'tags']
        for tag in tags:
            if tag not in tag2idx:
                tag2idx[tag] = tag_length
                tag_length += 1
    
    meta = {'song_length':song_length,
            'tag_length':tag_length,
            'tag2idx':tag2idx}
    
    print('Make pickle')
    with open(result_fname, 'wb') as fp:
        pkl.dump(meta, fp)

if __name__ == '__main__':
    fire.Fire(get_meta)
