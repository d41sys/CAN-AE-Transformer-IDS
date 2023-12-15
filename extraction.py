import pandas as pd
# import vaex
import numpy as np
import glob
import dask.dataframe as dd
import json
from sklearn.model_selection import train_test_split
import math
import csv
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix
import time
import _warnings
import tensorflow as tf
from tqdm import tqdm
import swifter
import argparse
import helper_functions
from importlib import reload
import sys
import os

def get_all_data(attack_dict):
    df_aggregation = []
    
    for attack_name, metadata in attack_dict.items():    
        if "accelerator" not in attack_name and "metadata" not in attack_name:
            file_name = '/home/tiendat/transformer-entropy-ids/road/attacks/{}.log'.format(attack_name)
            df_attack = helper_functions.make_can_df(file_name)
            df_attack = helper_functions.add_time_diff_per_aid_col(df_attack)
            df_aggregation.append(df_attack)
            print(f"Finish preprocess {file_name}")
    return df_aggregation

def get_time_interval(attack_dict):
    attack_metadata = []
    
    for attack_name, metadata in attack_dict.items():    
        if "accelerator" not in attack_name and "metadata" not in attack_name:
            print(f"Finish get time interval of {attack_name}")
            attack_metadata.append([tuple(attack_dict[attack_name]["injection_interval"])])
    return attack_metadata

def mark_label(df_aggregation, attack_metadata, attack_dict):
    count = 0
    for attack_name, metadata in attack_dict.items():    
        if "accelerator" not in attack_name and "metadata" not in attack_name:
            print(f"Index {count}: {attack_name} --- {attack_dict[attack_name]['injection_id']}")
            if attack_dict[attack_name]["injection_id"] != "XXX":
                df_aggregation[count] = helper_functions.add_actual_attack_col(df_aggregation[count], attack_metadata[count], int(attack_dict[attack_name]["injection_id"], 16), attack_dict[attack_name]["injection_data_str"], attack_name)
                print(len(df_aggregation[count][df_aggregation[count]['label'] == True]['label']))
            else:
                df_aggregation[count] = helper_functions.add_actual_attack_col(df_aggregation[count], attack_metadata[count], "XXX", attack_dict[attack_name]["injection_data_str"], attack_name)
                print(len(df_aggregation[count][df_aggregation[count]['label'] == True]['label']))
            count += 1
    return df_aggregation

def filter_attack(arr, keyword):
    sub_array = []
    for item in arr:
        if keyword in item:
            sub_array.append(item)
    return sub_array

def get_df_normal():
    df_normal = []
    for file_name in os.listdir("/home/tiendat/transformer-entropy-ids/road/ambient"):
        if "metadata" not in file_name:
            print(file_name)
            file_name = '/home/tiendat/transformer-entropy-ids/road/ambient/' + file_name
            df = helper_functions.make_can_df(file_name)
            df = helper_functions.add_time_diff_per_aid_col(df)
            df['label'] = [False] * df.shape[0]
            print(df.shape)
            df_normal.append(df)
    return df_normal

def get_df_attack():
    with open("/home/tiendat/transformer-entropy-ids/road/attacks/capture_metadata.json", "r") as read_file:
        attack_dict = json.load(read_file)

    df_attack = get_all_data(attack_dict)
    attack_metadata = get_time_interval(attack_dict)
    df_attack = mark_label(df_attack, attack_metadata, attack_dict) 
    return df_attack
        
def get_all():
    df_all = []
    df_normal = get_df_normal()
    df_attack = get_df_attack()
    df_all.append(df_normal)
    df_all.append(df_attack)
    
    return df_all

def split_sub_df(df, stride):
    sub_dfs = []
    for subd in df:
        # Loop through the original DataFrame in steps of 15 rows
        subd = subd.sort_values(by=['time'], ascending=False)
        print(len(subd))
        for i in range(0, len(subd), stride):
            if len(subd) - i < stride:
                # print(len(subd)-i)
                sub_df = subd.iloc[i:len(subd)-1]
            else:
                sub_df = subd.iloc[i:i+stride]
            sub_dfs.append(sub_df)
    return sub_dfs

def create_rule_dict(df_all, stride):
    rule_dict = {}
    sub_dfs = split_sub_df(df_all, stride)
    for subd in sub_dfs:
        list_entropy = subd.groupby('aid').apply(calculate_entropy)
    return rule_dict

def add_entropy_value(aid, entropy):
    if aid not in rule_dict:
        rule_dict[aid] = { 'max_entropy': entropy, 'min_entropy': entropy}
    else:
        max_entropy = max(rule_dict[aid]['max_entropy'], entropy)
        min_entropy = min(rule_dict[aid]['min_entropy'], entropy)
        rule_dict[aid]['max_entropy'] = max_entropy
        rule_dict[aid]['min_entropy'] = min_entropy

# Function to calculate entropy
def calculate_entropy(subd):
    data_column = subd['data']
    aid = subd['aid'].unique()[0]
    value_counts = data_column.value_counts()
    total_elements = len(data_column)
    probabilities = value_counts / total_elements
    entropy = -np.sum(probabilities * np.log2(probabilities))
    add_entropy_value(aid, entropy)
    return entropy


    
rule_dict = {int(k): v for k, v in rule_dict.items()}
with open("rules_30.json", "w") as outfile:
    json.dump(rule_dict, outfile)


# ATATACK ===========================================================
# FOR ATTACK DATAFRAMEs
sub_dfs = []
stride = 15

with open("rules.json", "r") as read_file:
    rule_dict = json.load(read_file)

for subd in df_attack:
    # Loop through the original DataFrame in steps of 15 rows
    subd = subd.sort_values(by=['time'], ascending=False)
    print(len(subd))
    for i in range(0, len(subd), stride):
        if len(subd) - i < stride:
            print(len(subd)-i)
            sub_df = subd.iloc[i:len(subd)-1]
        else:
            sub_df = subd.iloc[i:i+stride]
        sub_dfs.append(sub_df)
        

# Function to calculate entropy
def calculate_entropy_attack(subd):
    print(subd)
    data_column = subd['data']
    aid = str(subd['aid'].unique()[0])
    value_counts = data_column.value_counts()
    total_elements = len(data_column)   
    probabilities = value_counts / total_elements
    entropy = -np.sum(probabilities * np.log2(probabilities))
    predict = False
    if entropy > rule_dict[aid]['max_entropy']:
        predict = True
    elif entropy < rule_dict[aid]['min_entropy']:
        predict = False
    return predict

total = 0
attack_real = 0
attack_predict = 0

for subd in sub_dfs:
    total += len(subd)
    if True in subd['label'].unique():
        attack_real += 1
    predict_result = subd.groupby('aid').apply(calculate_entropy_attack)
    if predict_result.any():
        attack_predict += 1

print(f"Total: {total}")
print(f"Attack real: {attack_real}")
print(f"Attack predict: {attack_predict}")
print(f"Accuracy: {attack_predict/total}")
print(f"Recall: {attack_predict/attack_real}")
print(f"Precision: {attack_predict/(attack_predict + (total - attack_real))}")
print(f"F1 score: {2 * (attack_predict/(attack_predict + (total - attack_real))) * (attack_predict/attack_real) / ((attack_predict/(attack_predict + (total - attack_real))) + (attack_predict/attack_real))}")