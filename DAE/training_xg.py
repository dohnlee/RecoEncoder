#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle as pkl

import xgboost

from util import *

def xgb(args):
    
    params = {
        'objective': 'binary:logistic',
        'eta': 0.1,
        'booster': 'gbtree',
        'max_depth': 7,
        'nthread': 50,
        'seed': 1,
        'eval_metric': 'auc'
    }
    
    if args.device == 'gpu':
        params['gpu_id'] = 0
        params['tree_method'] = 'gpu_hist'

    i_fname_lst = os.listdir(args.xg_input_fname)
    init = True
    for i_fname in i_fname_lst:
        xgb_input = load_pickle(os.path.join(args.xg_input_fname, i_fname))
        xgtrain = xgboost.DMatrix(xgb_input[:,:-1], xgb_input[:,-1])
        if init:
            model = xgboost.train(
                params=list(params.items()),
                early_stopping_rounds=30,
                verbose_eval=10,
                dtrain=xgtrain,
                evals=[(xgtrain, 'train')],
                num_boost_round=300
            )
            model.save_model(args.xg_fname)
            init = False
        else:
            model = xgboost.train(
                params=list(params.items()),
                early_stopping_rounds=30,
                verbose_eval=10,
                dtrain=xgtrain,
                evals=[(xgtrain, 'train')],
                num_boost_round=300,
                xgb_model=args.xg_fname
            )
            model.save_model(args.xg_fname)

    write_pickle(model, args.xg_fname)


if __name__ == '__main__':
    xgb(
        xgb_input_fname='./res/feature/train/train_1_song_xgb_input.pkl',
        result_fname='./res/xgb/song_xgb_model_1.pkl'
    )
