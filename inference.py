# -*- coding: utf-8 -*-
import random
import argparse
import pickle as pkl
from util import *
from data import *
from network import *
from training_ae import *
from infer import *
from training_xg import xgb

parser = argparse.ArgumentParser(description='Inference : Task - Playlist Continuation')
parser.add_argument('--train_fname', metavar='DIR',
                    help='file name to traing data json [default: ./data/train.json]',
                    default='./data/train.json')
parser.add_argument('--infer_fname', metavar='DIR',
                    help='file name to inferencing data json [default: ./data/val.json]',
                    default='./data/val.json')
parser.add_argument('--results_fname', metavar='DIR',
                    help='file name to inferencing data json [default: ./res/results.json]',
                    default='./res/results.json')
parser.add_argument('--meta_fname', metavar ='DIR',
                    help='file name to meta data pickle [default: ./data/meta.pkl]',
                    default='./data/meta.pkl')
parser.add_argument('--xg_input_fname', metavar ='DIR',
                    help='file name to meta data pickle [default: ./data/tmp/]',
                    default='./data/tmp/')
parser.add_argument('--codict_fname', metavar ='DIR',
                    help='file name to cooccurrence pickle [default: ./data/codict.pkl]',
                    default='./data/codict.pkl')
parser.add_argument('--ae_fname', metavar='DIR',
                    help='file name to save trained candidate model [default: ./res/dae]',
                    default='./res/dae')
parser.add_argument('--xg_fname', metavar='DIR',
                    help='file name to save trained ranking model [default: ./res/xg]',
                    default='./res/xg')

# inferencing
inference = parser.add_argument_group('Inferencing options')
inference.add_argument('--mode', type=str, default='all',
                    help='you can choose ae, all [default=all]')
inference.add_argument('--have_meta', default=True,
                    help='exist meta pickle [default=True]')
inference.add_argument('--device', type=str, default='gpu',
                    help='cpu or gpu')

def main():
    args = parser.parse_args()
    if args.mode not in {'ae', 'all'}:
        sys.exit('check mode')
    print('start inference mode {}'.format(args.mode))
    pp = PreProcess()
     
    data = load_json(args.train_fname)
    if args.have_meta:
        meta = load_pickle(args.meta_fname)
    else:
        meta = pp.make_meta(data)
        write_pickle(meta, args.meta_fname) 
    infer = Inference(meta, args.device)
    
    test_data = load_json(args.infer_fname)

    print('make inputs ... ')
    trains = pp.make_input(data, meta)
    questions = pp.make_input(test_data, meta)
    
    token_length = len(meta['token2idx'])
    song_length = len(meta['song2idx'])
    tag_length = len(meta['tag2idx'])
    input_size = token_length + song_length + tag_length
    model = AutoEncoder(input_size=input_size)
    if args.device == 'gpu':
        model.cuda()
        model.load_state_dict(torch.load(args.ae_fname, map_location=torch.device('cuda')))
    else:
        model.load_state_dict(torch.load(args.ae_fname, map_location=torch.device('cpu')))
    if args.mode == 'ae':
        results = infer.inference(model, questions, 100, 10)
        write_json(results, args.results_fname)
    else:
        co_song, song_df = pp.make_codict(trains, meta)
        print('make candidate ... ')
        candidates, scores = infer.inference(model, questions, 2000, 10, with_score=True)
        print('Re-ranking ... ')
        results = infer.ranking(args, questions, candidates, scores, song_df, co_song)
        write_json(results, args.results_fname)
    print('Inference End')

if __name__ == '__main__':
   main() 
