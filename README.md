# DeepFakeDefenders
This is code by team fmsh_ck, ranked 6th in [Inclusion・The Global Multimedia Deepfake Detection](https://www.kaggle.com/competitions/multi-ffdi/overview) Track 2.

## Requirement

To install the dependencies run:
```bash
$ pip install -r requirements.txt
```

## Training

To train a model run:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --train_label trainset_label.txt --val_label valset_label.txt --save_path detectmodel
```
The trained model will be saved in ./exps/detectmodel/

The pretrained weights can be downloaded from [Baidu Pan](https://pan.baidu.com/s/1DxcsK1yKrA2Tuvi7Zf28rg?pwd=sw9j )

## Evaluation on valset

To evaluate on the valset run:
```bash
CUDA_VISIBLE_DEVICES=0 python val.py --model_path ./exps/detectmodel/epoch_0.model --test_path valset_label.txt --save_path exps/detectmodel/val_0.csv
```
The evaluation result will be saved in exps/detectmodel/val_0.csv

## Evaluation on testset

To evaluate on the testset run:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model_path ./exps/detectmodel/epoch_0.model --test_path testset1seen_nolabel.txt --save_path exps/detectmodel/infer_0.csv
```
The test result will be saved in exps/detectmodel/infer_0.csv

## Evaluation on one video
To evaluate on one video run:
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --model_path ./exps/detectmodel/epoch_0.model --test_path video_name.mp4
```
