import torch
import torch.nn as nn
import math
from torch_dataset import MyDatasetSLForTransDNNT
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix
import os
import time
import warnings
import torch.nn.functional as F

class Config:
    def __init__(self):
        self.model_name = 'Transformer'
        self.slide_window = 2
        self.slsum_count = int(math.pow(16, self.slide_window))  # 滑动窗口计数的特征的长度 n-gram?
        self.dnn_out_d = 8  # 经过DNN后的滑动窗口计数特征的维度 Dimensions of sliding window count features after DNN
        self.head_dnn_out_d = 32
        self.d_model = self.dnn_out_d + self.head_dnn_out_d  # transformer的输入的特征的维度, dnn_out_d + 包头长度 The dimension of the input feature of the transformer, dnn_out_d + header length
        self.pad_size = 100
        self.max_time_position = 10000
        self.nhead = 5
        self.num_layers = 3
        self.gran = 1e-6
        self.log_e = 2
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.classes_num = 3
        self.batch_size = 10
        self.epoch_num = 5
        self.lr = 0.001
        self.train_pro = 0.8  # 训练集比例 Ratio of training set

        self.data_root_dir = './data/car-hacking'
        self.sl_sum_dir = './data/car_hacking-data_slide_count_' + str(
            self.slide_window) + '_arr'
        self.time_dir = './data/car_hacking-data_time'
        self.names_file = './data/name_class_CICIDS_3.csv'
        self.model_save_path = './model/' + self.model_name + '/'
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        self.result_file = '/Users/d41sy/Desktop/sch/coding/ml-ids/result/trans8_performance.txt'

        self.isload_model = False  # 是否加载模型继续训练 Whether to load the model and continue training
        self.start_epoch = 24  # 加载的模型的epoch The epoch of the loaded model
        self.model_path = 'model/' + self.model_name + '/' + self.model_name + '_model_' + str(self.start_epoch) + '.pth'  # 要使用的模型的路径 path to the model to use

config = Config()

fin = open(config.result_file, 'a')
fin.write('-------------------------------------\n')
fin.write(config.model_name + '\n')
fin.write('begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + '\n')
fin.write('data root dir: ' + config.data_root_dir + '\n')
fin.write('sl_sum_dir: ' + config.sl_sum_dir + '\n')
fin.write('names_file: ' + config.names_file + '\n')
fin.write('d_model: ' + str(config.d_model) + '\t pad_size: ' + str(config.pad_size) + '\t nhead: ' + str(config.nhead)
          + '\t num_layers: ' + str(config.num_layers) + '\t head_dnn_out_d: '+ str(config.head_dnn_out_d) +'\n')
fin.write(
    'batch_size: ' + str(config.batch_size) + '\t train pro: ' + str(config.train_pro) + '\t learning rate: ' + str(
        config.lr) + '\n\n')
fin.close()

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset = MyDatasetSLForTransDNNT(config.data_root_dir, config.sl_sum_dir, config.time_dir, config.names_file, config.pad_size, config.d_model, config.max_time_position, config.gran, config.log_e)
size = len(dataset)
print(dataset.__getitem__(0))

train_size = int(config.train_pro * size)
test_size = size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size)
print('finish load data')