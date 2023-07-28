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

warnings.filterwarnings("ignore")

NUM_CLASSES = 5

def draw_confusion(label_y, pre_y, path):
    confusion = confusion_matrix(label_y, pre_y)
    print(confusion)

def write_result(fin, label_y, pre_y, classes_num):
    if classes_num > 2:
        accuracy = accuracy_score(label_y, pre_y)
        macro_precision = precision_score(label_y, pre_y, average='macro')
        macro_recall = recall_score(label_y, pre_y, average='macro')
        macro_f1 = f1_score(label_y, pre_y, average='macro')
        micro_precision = precision_score(label_y, pre_y, average='micro')
        micro_recall = recall_score(label_y, pre_y, average='micro')
        micro_f1 = f1_score(label_y, pre_y, average='micro')
        print('  -- test result: ')
        fin.write('  -- test result: \n')
        print('    -- accuracy: ', accuracy)
        fin.write('    -- accuracy: ' + str(accuracy) + '\n')
        print('    -- macro precision: ', macro_precision)
        fin.write('    -- macro precision: ' + str(macro_precision) + '\n')
        print('    -- macro recall: ', macro_recall)
        fin.write('    -- macro recall: ' + str(macro_recall) + '\n')
        print('    -- macro f1 score: ', macro_f1)
        fin.write('    -- macro f1 score: ' + str(macro_f1) + '\n')
        print('    -- micro precision: ', micro_precision)
        fin.write('    -- micro precision: ' + str(micro_precision) + '\n')
        print('    -- micro recall: ', micro_recall)
        fin.write('    -- micro recall: ' + str(micro_recall) + '\n')
        print('    -- micro f1 score: ', micro_f1)
        fin.write('    -- micro f1 score: ' + str(micro_f1) + '\n\n')
        report = classification_report(label_y, pre_y)
        fin.write(report)
        fin.write('\n\n')
    else:
        accuracy = accuracy_score(label_y, pre_y)
        precision = precision_score(label_y, pre_y)
        recall = recall_score(label_y, pre_y)
        f1 = f1_score(label_y, pre_y)
        print('  -- test result: ')
        print('    -- accuracy: ', accuracy)
        fin.write('    -- accuracy: ' + str(accuracy) + '\n')
        print('    -- recall: ', recall)
        fin.write('    -- recall: ' + str(recall) + '\n')
        print('    -- precision: ', precision)
        fin.write('    -- precision: ' + str(precision) + '\n')
        print('    -- f1 score: ', f1)
        fin.write('    -- f1 score: ' + str(f1) + '\n\n')
        report = classification_report(label_y, pre_y)
        fin.write(report)
        fin.write('\n\n')

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
        self.lr = 0.0001 #0.00001 learning rate 
        self.train_pro = 0.7  # 训练集比例 Ratio of training set

        self.root_dir = './data/Processed/TFRecord_w29_s29/1/'
        self.model_save_path = './model/' + self.model_name + '/'
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        self.result_file = '/mnt/hdd2/transformer-entropy-ids/result/trans8_performance.txt'

        self.isload_model = False  # 是否加载模型继续训练 Whether to load the model and continue training
        self.start_epoch = 24  # 加载的模型的epoch The epoch of the loaded model
        self.model_path = 'model/' + self.model_name + '/' + self.model_name + '_model_' + str(self.start_epoch) + '.pth'  # 要使用的模型的路径 path to the model to use

class DNN(nn.Module):
    def __init__(self, d_in, d_out):  # config.slsum_count, config.dnn_out_d
        super(DNN, self).__init__()
        self.l1 = nn.Linear(d_in, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, d_out)

    def forward(self, x):
        # print('x: ', x.numpy()[0])
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        # print('dnn out: ', out.detach().numpy()[0])
        return out

class Time_Positional_Encoding(nn.Module):
    def __init__(self, embed, max_time_position, device):
        super(Time_Positional_Encoding, self).__init__()
        self.device = device

    def forward(self, x, time_position):
        out = x.permute(1, 0, 2)
        out = out + nn.Parameter(time_position, requires_grad=False).to(self.device)
        out = out.permute(1, 0, 2)
        return out

class MyTrans(nn.Module):
    def __init__(self, config):
        super(MyTrans, self).__init__()
        self.dnn = DNN(config.slsum_count, config.dnn_out_d).to(config.device)
        self.head_dnn = DNN(4, config.head_dnn_out_d).to(config.device)
        self.position_embedding = Time_Positional_Encoding(config.d_model, config.max_time_position, config.device).to(config.device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.nhead).to(config.device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config.num_layers).to(config.device)
        self.fc = nn.Linear(config.d_model, config.classes_num).to(config.device)
        self.pad_size = config.pad_size
        self.dnn_out_d = config.dnn_out_d
        self.head_dnn_out_d = config.head_dnn_out_d

    def forward(self, header, sl_sum, mask, time_position):
        dnn_out = torch.empty((sl_sum.shape[0], self.dnn_out_d, 0)).to(config.device)

        for i in range(self.pad_size):
            tmp = self.dnn(sl_sum[:, i, :]).unsqueeze(2)
            dnn_out = torch.concat((dnn_out, tmp), dim=2)
        dnn_out = dnn_out.permute(0, 2, 1)

        head_dnn_out = torch.empty((header.shape[0], self.head_dnn_out_d, 0)).to(config.device)
        for i in range(self.pad_size):
            tmp = self.head_dnn(header[:, i, :]).unsqueeze(2)
            head_dnn_out = torch.concat((head_dnn_out, tmp), dim=2)
        head_dnn_out = head_dnn_out.permute(0, 2, 1)

        x = torch.concat((head_dnn_out, dnn_out), dim=2).permute(1, 0, 2)

        out = self.position_embedding(x, time_position)
        out = self.transformer_encoder(out, src_key_padding_mask=mask)
        out = out.permute(1, 0, 2)
        out = torch.sum(out, 1)
        out = self.fc(out)
        return out

config = Config()
print(config.device)

fin = open(config.result_file, 'a')
fin.write('-------------------------------------\n')
fin.write(config.model_name + '\n')
fin.write('begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + '\n')
fin.write('data root dir: ' + config.root_dir + '\n')
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

train_dataset = DatasetPreprocess(config.root_dir, config.window_size, config.pad_size, config.d_model, config.max_time_position, config.gran, config.log_e, is_train=True)
test_dataset = DatasetPreprocess(config.root_dir, config.window_size, config.pad_size, config.d_model, config.max_time_position, config.gran, config.log_e, is_train=False)

print("TRAIN SIZE:", len(train_dataset), " TEST SIZE:", len(test_dataset), " SIZE:", len(train_dataset)+len(test_dataset), " TRAIN RATIO:", round(len(train_dataset)/(len(train_dataset)+len(test_dataset))*100), "%")
# print("TRAIN DATA:", len(train_dataset[0]['header']))


# 2 DataLoader? 
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size)
print('finish load data')

if config.isload_model:
    print("Case loaded")
    fin = open(config.result_file, 'a')
    fin.write('load trained model :    model_path: ' + config.model_path)
    model = torch.load(config.model_path)
    start_epoch = config.start_epoch
    fin.close()
else:
    print("Case trained")
    model = MyTrans(config)
    start_epoch = -1
loss_func = nn.CrossEntropyLoss().to(config.device)
opt = torch.optim.Adam(model.parameters(), lr=config.lr)

for epoch in range(start_epoch + 1, config.epoch_num):
    fin = open(config.result_file, 'a')
    print('--- epoch ', epoch)
    fin.write('-- epoch ' + str(epoch) + '\n')
    for i, sample_batch in enumerate(train_loader):
        batch_header = sample_batch['header'].type(torch.FloatTensor).to(config.device)
        batch_sl_sum = sample_batch['payload'].type(torch.FloatTensor).to(config.device)
        batch_mask = sample_batch['mask'].to(config.device)
        batch_label = sample_batch['label'].to(config.device)
        batch_time_position = sample_batch['time'].to(config.device)
        # print("BATCH HEADER: ", len(batch_header[0]), " \nBATCH PAYLOAD: ", batch_sl_sum[0][0]," \nBATCH MASK: ", len(batch_mask)," \nBATCH LABEL: ", batch_label," \nBATCH TIME POSITION: ", batch_time_position[0])
        out = model(batch_header, batch_sl_sum, batch_mask, batch_time_position)
        loss = loss_func(out, batch_label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 20 == 0:
            print('iter {} loss: '.format(i), loss.item())
    torch.save(model, (config.model_save_path + config.model_name + '_model_{}.pth').format(epoch))

    # test
    label_y = []
    pre_y = []
    with torch.no_grad():
        for j, test_sample_batch in enumerate(test_loader):
            test_header = test_sample_batch['header'].type(torch.FloatTensor).to(config.device)
            test_sl_sum = test_sample_batch['payload'].type(torch.FloatTensor).to(config.device)
            test_mask = test_sample_batch['mask'].to(config.device)
            test_label = test_sample_batch['label'].to(config.device)
            test_time_position = test_sample_batch['time'].to(config.device)
            
            test_out = model(test_header, test_sl_sum, test_mask, test_time_position)

            pre = torch.max(test_out, 1)[1].cpu().numpy()
            pre_y = np.concatenate([pre_y, pre], 0)
            label_y = np.concatenate([label_y, test_label.cpu().numpy()], 0)
        write_result(fin, label_y, pre_y, config.classes_num)
        draw_confusion(label_y, pre_y, '')
    fin.close()

fin = open(config.result_file, 'a')
fin.write('\n\n\n')
fin.close()