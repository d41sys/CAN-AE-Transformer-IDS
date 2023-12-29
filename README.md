# CAN-AE-Transformer-IDS
This is source code for paper

## Data Preprocessing
### Car Hacking dataset
```Python
CUDA_VISIBLE_DEVICES=0 python3 splitDataIntoSession.py --window_size=29 --strided=29 --attack_type=chd > data_preprocessing_chd.txt
```
### ROAD Fabrication dataset
```Python
CUDA_VISIBLE_DEVICES=0 python3 splitDataIntoSession.py --window_size=15 --strided=15 --attack_type=road_fab --indir=./road/fab_dataset --outdir=./road/fab_multi/TFRecord > data_preprocessing_roadfab.txt
```
### ROAD Masquerade dataset
```Python
CUDA_VISIBLE_DEVICES=0 python3 splitDataIntoSession.py --window_size=15 --strided=15 --attack_type=road_mas --indir=./road/mas_dataset --outdir=./road/mas_multi/TFRecord > data_preprocessing_roadmas.txt
```

## Train test split
```Python
CUDA_VISIBLE_DEVICES=1 python3 trainTestSplit.py --data_path=./data/CHD --window_size 29 --strided 29 --rid 1
```
```Python
CUDA_VISIBLE_DEVICES=1 python3 trainTestSplit.py --data_path=./data/fab_dataset --window_size 15 --strided 15 --rid 1
```
```Python
CUDA_VISIBLE_DEVICES=1 python3 trainTestSplit.py --data_path=./data/mas_dataset --window_size 15 --strided 15 --rid 1
```

## Train

```Python
CUDA_VISIBLE_DEVICES=2 python3 new_trainer.py --indir=./road/mas_multi/TFRecord_w15_s15/1/ --window_size=15 --batch_size=32 --type=road_mas --mode=cb --tse=True --epoch=150 --ver=2
```

```Python
CUDA_VISIBLE_DEVICES=2 python3 new_trainer.py --indir=./road/fab_multi/TFRecord_w15_s15/1/ --window_size=15 --batch_size=32 --type=road_fab --mode=cb --tse=True --epoch=300 --ver=2
```