# RecoEncoder
AutoEncoder for Recommendation

written by g-lab

## Introduction
카카오 아레나 Melon Playlist Continuation
https://arena.kakao.com/c/7

모델은 두가지로 구성되어집니다. 
1. Candidate Model - Denoising AutoEncoder
2. Ranking Model - xgboost

## Candidate Model - Denoising AutoEncoder
<img src="./images/DAE.png" width="60%" height="60%">

- 추천을 하기 위한 seed song/tag 가 없는 경우 _cold start problem_ 을 해결하기 위해 playlist title을 tokenize한 후에 input으로 사용했습니다.
- 오토인코더의 성능을 높이기 위해, 일부 곡이나 태그들을 가리고 복원하는 과정에서 맞추도록 Denoising AutoEncoder를 사용하였습니다.

__Reference__
- [Training Deep AutoEncoders for Collaborative Filtering](https://arxiv.org/abs/1708.01715)
- [MMCF: Multimodal Collaborative Filtering for Automatic Playlist Continuation](https://dl.acm.org/doi/10.1145/3267471.3267482)

## Ranking Model - Xgboost


## Requirement

다음과 같은 환경에서 실험을 진행하였습니다.

| 디바이스 / 패키지 | 정보 / 버전 |
| :----: | :----: |
| **CPU** | Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz |
| **GPU** | GeForce GTX 1080Ti |
| **OS** | Ubuntu 16.04.6 |
| **CUDA** | 10.0, V10.0.130 |
| **python** | 3.7.4 |
| **numpy** | 1.17.2 |
| **torch** | 1.5.1+cu101 |
| **xgboost** | 1.1.1 |
| **khaiii** | 0.4 |
| **tqdm** | 4.36.1 |


## Dataset Format
https://arena.kakao.com/c/7/data
```
{'tags': ['락'],
 'id': 61281,
 'plylst_title': '여행같은 음악',
 'songs': [525514,
  129701,
  383374,
  562083,
  297861,
  139541,
  351214,
  650298,
  531057,
  205238,
  706183,
  127099,
  660493,
  461973,
  121455,
  72552,
  223955,
  324992,
  50104],
 'like_cnt': 71,
 'updt_date': '2013-12-19 18:36:19.000'}
```
## Train
`python train.py -h` 을 실행하면 다음과 같이 train에 필요한 옵션들을 확인할 수 있습니다.

```
usage: train.py [-h] [--train_fname DIR] [--meta_fname DIR]
                [--xg_input_fname DIR] [--codict_fname DIR] [--ae_fname DIR]
                [--xg_fname DIR] [--mode MODE] [--have_meta HAVE_META]
                [--validate VALIDATE] [--lr LR] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--drop_prob DROP_PROB]
                [--noise_prob NOISE_PROB] [--milestones MILESTONES]
                [--gamma GAMMA] [--check_point CHECK_POINT] [--device DEVICE]

Train Candidate Model(AE) and Train Rank Model(XGBoost)

optional arguments:
  -h, --help            show this help message and exit
  --train_fname DIR     file name to traing data json [default:
                        ./data/train.json]
  --meta_fname DIR      file name to meta data pickle [default:
                        ./data/meta.pkl]
  --xg_input_fname DIR  file name to meta data pickle [default: ./data/tmp/]
  --codict_fname DIR    file name to cooccurrence pickle [default:
                        ./data/codict.pkl]
  --ae_fname DIR        file name to save trained candidate model [default:
                        ./res/dae]
  --xg_fname DIR        file name to save trained ranking model [default:
                        ./res/xg]

Learning options:
  --mode MODE           you can choose ae / xgboost [default=ae]
  --have_meta HAVE_META
                        exist meta pickle [default=False]
  --validate VALIDATE   validate dataset and check NDCG [default=False]
  --lr LR               initial learning rate [default: 0.0005]
  --epochs EPOCHS       number of epochs for training [default: 300]
  --batch_size BATCH_SIZE
                        batch size for training [default:128]
  --drop_prob DROP_PROB
                        dropout rate for training [default:0.]
  --noise_prob NOISE_PROB
                        denoising probability for training [default:0.8]
  --milestones MILESTONES
                        for multi step lr decay [default:[50, 100, 150, 200,
                        250]]
  --gamma GAMMA         for multi step lr decay [default:0.1]
  --check_point CHECK_POINT
                        chek point fot save model [default:50]
  --device DEVICE       cpu or gpu
```
## Inference
`train.py`와 마찮가지로 `python inference.py -h` 을 실행하면 다음과 같이 inference에 필요한 옵션들을 확인할 수 있습니다.


```
usage: inference.py [-h] [--train_fname DIR] [--infer_fname DIR]
                    [--results_fname DIR] [--meta_fname DIR]
                    [--xg_input_fname DIR] [--codict_fname DIR]
                    [--ae_fname DIR] [--xg_fname DIR] [--mode MODE]
                    [--have_meta HAVE_META] [--device DEVICE]

Inference : Task - Playlist Continuation

optional arguments:
  -h, --help            show this help message and exit
  --train_fname DIR     file name to traing data json [default:
                        ./data/train.json]
  --infer_fname DIR     file name to inferencing data json [default:
                        ./data/val.json]
  --results_fname DIR   file name to inferencing data json [default:
                        ./res/results.json]
  --meta_fname DIR      file name to meta data pickle [default:
                        ./data/meta.pkl]
  --xg_input_fname DIR  file name to meta data pickle [default: ./data/tmp/]
  --codict_fname DIR    file name to cooccurrence pickle [default:
                        ./data/codict.pkl]
  --ae_fname DIR        file name to save trained candidate model [default:
                        ./res/dae]
  --xg_fname DIR        file name to save trained ranking model [default:
                        ./res/xg]

Inferencing options:
  --mode MODE           you can choose ae, all [default=all]
  --have_meta HAVE_META
                        exist meta pickle [default=True]
  --device DEVICE       cpu or gpu
```
