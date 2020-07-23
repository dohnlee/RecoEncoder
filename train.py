# -*- coding: utf-8 -*-
import random
import argparse
import pickle as pkl
import torch
from util import *
from DAE.data import *
from DAE.network import *
from DAE.training_ae import *
from DAE.training_xg import xgb

# data
parser = argparse.ArgumentParser(description='Train Candidate Model(AE) and Train Rank Model(XGBoost)')
parser.add_argument('--train_fname', metavar='DIR',
                    help='file name to traing data json [default: ./data/train.json]',
                    default='./data/train.json')
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

# learning
learn = parser.add_argument_group('Learning options')
learn.add_argument('--mode', type=str, default='ae',
                    help='you can choose ae / xgboost [default=ae]')
learn.add_argument('--have_meta', default=False,
                    help='exist meta pickle [default=False]')
learn.add_argument('--val_ratio', type=float, default=0.,
                    help='validation dataset ratio for check NDCG [default=0]')
learn.add_argument('--lr', type=float, default=0.0005,
                    help='initial learning rate [default: 0.0005]')
learn.add_argument('--epochs', type=int, default=300,
                    help='number of epochs for training [default: 300]')
learn.add_argument('--batch_size', type=int, default=128,
                    help='batch size for training [default:128]')
learn.add_argument('--drop_prob', type=float, default=0.,
                    help='dropout rate for training [default:0.]')
learn.add_argument('--noise_prob', type=float, default=0.8,
                    help='denoising probability for training [default:0.8]')
learn.add_argument('--milestones', type=list, default=[50, 100, 150, 200, 250],
                    help='for multi step lr decay [default:[50, 100, 150, 200, 250]]')
learn.add_argument('--gamma', type=float, default=0.1,
                    help='for multi step lr decay [default:0.1]')
learn.add_argument('--check_point', type=int, default=50,
                    help='chek point fot save model [default:50]')
learn.add_argument('--device', type=str, default='gpu',
                    help='cpu or gpu')

def main():
    random.seed(777)
    args = parser.parse_args()
    print('start train mode {}'.format(args.mode))
    pp = PreProcess()
    splitter = ArenaSplitter()
     
    data = load_json(args.train_fname)
    if args.have_meta:
        meta = load_pickle(args.meta_fname)
    else:
        meta = pp.make_meta(data)
        write_pickle(meta, args.meta_fname) 
    infer = Inference(meta, args.device)
    
    print('make inputs ... ')
    trains = pp.make_input(data, meta)
    if args.mode not in {'ae', 'xgboost'}:
        sys.exit('Error : check mode --mode ae/xgboost')

    if args.mode != 'xgboost':
        train_ae = TrainAE(meta, args.device)
        if args.val_ratio > 0.: 
            random.shuffle(trains)
            trains, vals = splitter.split_data(trains, args.val_ratio)
            print('validation set')
            vals = splitter.mask_data(vals)
            model = train_ae.main(args, args.ae_fname, trains, vals)
        else:
            model = train_ae.main(args, args.ae_fname, trains)

    if args.mode != 'ae':
        train_ae = TrainAE(meta, args.device)
        # data split 5/5
        train_c, train_r = splitter.split_data(trains, 0.5)
        questions, answers = splitter._mask(train_r, ['songs', 'tags'], [])
        # train ae for ranking model
        model = train_ae.main(args, './res/ae_for_xg', train_c) 
        co_song, song_df = pp.make_codict(questions, meta)
        print('make candidate ... ')
        candidates, scores = infer.inference(model, questions, 200, 10, with_score=True)
        print('make xg inputs ... ')
        xgb_input = pp.make_xgboost_input(args,
                questions,
                candidates,
                scores,
                co_song,
                song_df,
                answers)
        xgb(args)
    print('Train End')

if __name__ == '__main__':
   main() 
