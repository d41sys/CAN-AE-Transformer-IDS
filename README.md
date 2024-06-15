# CAN-AE-Transformer-IDS
This is source code for paper "Multi-Classification In-Vehicle Intrusion Detection System using Packet- and Sequence-Level Characteristics from Time-Embedded Transformer with Autoencoder"

Journal: Knowledge-Based Systems

Link: https://doi.org/10.1016/j.knosys.2024.112091

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
### Car hacking dataset
```Python
CUDA_VISIBLE_DEVICES=1 python3 trainTestSplit.py --data_path=./data/CHD --window_size 29 --strided 29 --rid 1
```
### ROAD dataset
```Python
CUDA_VISIBLE_DEVICES=1 python3 trainTestSplit.py --data_path=./data/fab_dataset --window_size 15 --strided 15 --rid 1
```
```Python
CUDA_VISIBLE_DEVICES=1 python3 trainTestSplit.py --data_path=./data/mas_dataset --window_size 15 --strided 15 --rid 1
```

## Train
### ROAD masquerade dataset
```Python
CUDA_VISIBLE_DEVICES=2 python3 new_trainer.py --indir=./road/mas_multi/TFRecord_w15_s15/1/ --window_size=15 --batch_size=32 --type=road_mas --mode=cb --tse=True --epoch=150 --ver=2
```

### ROAD fabrication dataset
```Python
CUDA_VISIBLE_DEVICES=2 python3 new_trainer.py --indir=./road/fab_multi/TFRecord_w15_s15/1/ --window_size=15 --batch_size=32 --type=road_fab --mode=cb --tse=True --epoch=300 --ver=2
```

## Params
--indir, type=str          | Data path input

--window_size, type=int    | Window size (15 for ROAD and 29 for CHD)

--type, type=str           | Dataset type: `chd` or `road`

--epoch, type=int          | Epoch value

--batch_size, type=int     | Batch size value

--lr, type=float           | Learning rate

--mode, type=str           | `cb` or `ae` (the ways of using AE)

--tse, type=bool           | Using tse 

--ver, type=str            | Version value
