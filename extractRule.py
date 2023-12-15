import torch
import torch.nn as nn
import math
from dataPreprocess import DatasetPreprocess
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix
import os
import time
import warnings
import torch.nn.functional as F
import time
import json


class Config:
    def __init__(self):
        self.model_name = 'Transformer'
        self.slide_window = 1
        self.slsum_count = 8 #int(math.pow(16, self.slide_window))  # 滑动窗口计数的特征的长度 n-gram?
        self.dnn_out_d = 8 # 经过DNN后的滑动窗口计数特征的维度 Dimensions of sliding window count features after DNN 8
        self.head_dnn_out_d = 32 
        self.d_model = self.dnn_out_d + self.head_dnn_out_d # transformer的输入的特征的维度, dnn_out_d + 包头长度 The dimension of the input feature of the transformer, dnn_out_d + header length
        self.pad_size = 29
        self.window_size = 29
        self.max_time_position = 10000
        self.nhead = 5 # ori: 5
        self.num_layers = 5
        self.gran = 1e-6
        self.log_e = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_num = 5 
        self.batch_size = 10
        self.epoch_num = 20
        self.lr = 0.001 #0.00001 learning rate 
        self.train_pro = 0.7  # 训练集比例 Ratio of training set

        self.root_dir = './road/timesmooth/TFRecord_w29_s29/1/'
        self.model_save_path = './model/' + self.model_name + '/'
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        self.result_file = '/mnt/hdd2/transformer-entropy-ids/result/trans8_performance.txt'

        self.isload_model = False  # 是否加载模型继续训练 Whether to load the model and continue training
        self.start_epoch = 18  # 加载的模型的epoch The epoch of the loaded model
        # self.model_path = '../model/' + self.model_name + '/' + self.model_name + '_model_' + str(self.start_epoch) + '.pth'  # 要使用的模型的路径 path to the model to use
        self.model_path = './model/Transformer/best_model.pth'


def hamming_distance_int_list(seq1, seq2):
    # Calculate the Hamming distance between two binary sequences represented as lists
    return sum(bin(x ^ y).count('1') for x, y in zip(seq1, seq2))

def all_pair_hamming_distance_int_list(sequence_list):
    
    n = len(sequence_list)
    hamming_distances = {}
    max_distance = 0
    min_distance = 64

    for i in range(n):
        for j in range(i + 1, n):
            distance = hamming_distance_int_list(sequence_list[i].tolist(), sequence_list[j].tolist())
            max_distance = max(max_distance, distance)
            min_distance = min(min_distance, distance)
            hamming_distances[(i, j)] = distance
            # Since Hamming distance is symmetric (H(A, B) == H(B, A)),
            # we can also store the distance for (j, i) pair.
            hamming_distances[(j, i)] = distance
    
    return hamming_distances, max_distance, min_distance

config = Config()

train_dataset = DatasetPreprocess(config.root_dir, config.window_size, config.pad_size, config.d_model, config.max_time_position, config.gran, config.log_e, is_train=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

test_dataset = DatasetPreprocess(config.root_dir, config.window_size, config.pad_size, config.d_model, config.max_time_position, config.gran, config.log_e, is_train=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size)
print('finish load data')

linked_dict = {}

with torch.no_grad():
    for j, train_batch in enumerate(train_loader):
        # print(f"Origin time: {train_batch['oritime']} and length: {len(train_batch['oritime'])}")
        # aa
        canid = []
        message = []
        time_stamp = [] 
        
        for index in range(len(train_batch['oritime'])): 
            time_stamp.extend(train_batch['oritime'][index].tolist())
        
            for msg in train_batch['header'][index]:
                # print(f"{msg}")
                canid.append(''.join(map(str, msg.numpy())))
                # print(f"{''.join(map(str, msg))}")
                
            for msg in train_batch['payload'][index]:
                # print(f"{msg}")
                message.append(msg.numpy())
                # print(f"{''.join(map(str, msg))}")
        # print(canid)
        # print(len(message))
        # print(len(time_stamp))
        
        # Populate the dictionary
        for i, id in enumerate(canid):
            if id in linked_dict:
                linked_dict[id]['payload'].append(message[i])
                linked_dict[id]['timestamp'].append(time_stamp[i])
            else:
                linked_dict[id] = {'payload': [message[i]], 'timestamp': [time_stamp[i]]}

print('finish extract raw data')
            
for can_id in linked_dict:
    sorted_indices = sorted(range(len(linked_dict[can_id]['timestamp'])), key=lambda k: linked_dict[can_id]['timestamp'][k])
    linked_dict[can_id]['timestamp'] = [linked_dict[can_id]['timestamp'][i] for i in sorted_indices]
    linked_dict[can_id]['payload'] = [linked_dict[can_id]['payload'][i] for i in sorted_indices]

print('finish sort')

for can_id in linked_dict:
    time_diffs = [linked_dict[can_id]['timestamp'][i+1] - linked_dict[can_id]['timestamp'][i] for i in range(len(linked_dict[can_id]['timestamp']) - 1)]
    max_time_diff = max(time_diffs)
    min_time_diff = min(time_diffs)
    linked_dict[can_id]['max_time'] = max_time_diff
    linked_dict[can_id]['min_time'] = min_time_diff
    
print('finish extract max min time interval data')
    
for can_id in linked_dict:
    pair_distances, max_distance, min_distance = all_pair_hamming_distance_int_list(linked_dict[can_id]['payload'])
    # print(f"CANID: {can_id} with max: {linked_dict[can_id]['max_time']} and min: {linked_dict[can_id]['min_time']}")
    linked_dict[can_id]['max_hmc'] = max_distance
    linked_dict[can_id]['min_hmc'] = min_distance
    # for pair, distance in pair_distances.items():
    #     idx1, idx2 = pair
    #     print(f"Hamming distance between {linked_dict[can_id]['payload'][idx1]} and {linked_dict[can_id]['payload'][idx2]}: {distance}")
    print(f"CANID: {can_id} max distance: {max_distance} and min distance: {min_distance}")
    
print('finish extract max min hamming distance data')

with open('data.json', 'w') as file:
    json.dump(linked_dict, file)
