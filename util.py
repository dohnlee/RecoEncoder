# -*- coding: utf-8 -*-
import json
import pickle as pkl
import numpy as np

def write_json(data, fname):
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def load_json(fname):
    with open(fname) as f:
        json_obj = json.load(f)

    return json_obj

def write_pickle(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f) 

def load_pickle(path):
    with open(path, 'rb') as f:
        result = pkl.load(f)
    return result

def debug_json(r):
    print(json.dumps(r, ensure_ascii=False, indent=4))

def remove_seen(seen, l):
    seen = set(seen)
    return [x for x in l if not (x in seen)]

class Ndcg:
    def _idcg(self, I):
        return sum((1.0 / np.log(i+2) for i in range(I)))

    def __init__(self, answers):
        self._idcgs = [self._idcg(i) for i in range(101)]
        self.a_dict = {a['id']:a for a in answers}
    
    def _ndcg(self, gt, rec):
        dcg = 0
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0/np.log(i+2)
        return dcg / self._idcgs[len(gt)]

    def main(self, results):
        length = 0
        music_ndcg = 0
        tag_ndcg = 0
        for res in results:
            length += 1
            _id = res['id']
            answer = self.a_dict[_id]
            music_ndcg += self._ndcg(answer['songs'], res['songs'][:100])
            tag_ndcg += self._ndcg(answer['tags'], res['tags'][:10])

        music_ndcg /= length
        tag_ndcg /= length
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

