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