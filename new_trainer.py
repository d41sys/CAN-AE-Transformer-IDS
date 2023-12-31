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
import argparse
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings("ignore")

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
    def __init__(self, args):
        self.model_name = 'IDS-Transformer_' + args.type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.payload_size = 8 
        elif args.mode == "cb":
            self.mode = args.mode
            self.dout_mess = 10
            self.d_model = self.dout_mess
            self.nhead = 10 # ori: 5
        else:
            self.mode = args.mode
            self.dout_mess = 12
            self.d_model = self.dout_mess
            self.nhead = 6 # ori: 5
        
        self.tse = args.tse
        self.pad_size = args.window_size # 15
        self.window_size =  args.window_size # 15
        self.max_time_position = 10000
        self.num_layers = 6
        self.gran = 1e-7 # ori: 1e-6
        self.log_e = 2
        
        if args.type == 'chd':
            self.classes_num = 5
        elif args.type == 'road_mas':
            self.classes_num = 6
        else: # road_fab
            self.classes_num = 7 
        
            
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.lr = args.lr #0.0001 learning rate 
        self.root_dir = args.indir
        
        # self.root_dir = './data/Processed/TFRecord_w29_s29/2/'
        # self.root_dir = './road/predict_fab_multi/TFRecord_w15_s15/1/'
        # self.root_dir = './road/predict_mas/TFRecord_w15_s15/4/'
        self.model_save_path = './model/' + self.model_name + '/'
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        self.result_file = '/home/tiendat/transformer-entropy-ids/result/'+'pIDS_' + args.type + args.ver + '_' + args.mode + '.txt'

        self.isload_model = False  
        self.start_epoch = 24  # The epoch of the loaded model
        self.model_path = 'model/' + self.model_name + '/' + self.model_name + '_model_' + str(self.start_epoch) + '.pth' 
        
    def forward(self, x):
        #print('x: ', x.cpu().numpy()[0])
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        #print('dnn out: ', out.cpu().detach().numpy()[0])
        return out
    
class Autoencoder1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Autoencoder1D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return F.relu(x)
    
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class Time_Positional_Encoding(nn.Module):
    def __init__(self, embed, max_time_position, device):
        super(Time_Positional_Encoding, self).__init__()
        self.device = device

    def forward(self, x, time_position):
        out = x.permute(1, 0, 2)
        out = out + nn.Parameter(time_position, requires_grad=False).to(self.device)
        out = out.permute(1, 0, 2)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerPredictor(nn.Module):
    def __init__(self, config):
        super(TransformerPredictor, self).__init__()
        
        if config.mode == "ae":
            self.payload_ae = Autoencoder1D(config.payload_size, config.dout_payload).to(config.device)
            self.header_ae = Autoencoder1D(4, config.dout_header).to(config.device)
            self.dout_payload = config.dout_payload
            self.dout_header = config.dout_header
        elif config.mode == "cb":
            self.ae = Autoencoder1D(12, config.dout_mess).to(config.device)
            self.dout_mess = config.dout_mess
        
        self.mode = config.mode
        self.pad_size = config.pad_size
        self.tse = config.tse
        if config.tse == True:
            self.position_embedding = Time_Positional_Encoding(config.d_model, config.max_time_position, config.device).to(config.device)
            print("Got TSE")
        else:
            self.position_embedding = PositionalEncoding(config.d_model, dropout=0.0, max_len=config.max_time_position).to(config.device)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.nhead).to(config.device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config.num_layers).to(config.device)
        self.fc = nn.Linear(config.d_model, config.classes_num).to(config.device)
        print(f"Initial model with mode: {self.mode} \
                TSE: {self.tse}")
        
    def forward(self, header, sl_sum, mask, time_position):
        elif self.mode == "cb":
            x = torch.concat((header, sl_sum), dim=-1)
            ae_out = torch.empty((x.shape[0], 10, 0)).to(config.device)
            for i in range(self.pad_size):
                tmp = self.ae(x[:, i, :]).unsqueeze(2)
                ae_out = torch.concat((ae_out, tmp), dim=2)
            x = ae_out.permute(2, 0, 1)
        else:
            x = torch.concat((header, sl_sum), dim=-1).permute(1, 0, 2)
        
        if self.tse == True:
            out = self.position_embedding(x, time_position)
        else:
            out = self.position_embedding(x)
        out = self.transformer_encoder(out, src_key_padding_mask=mask)
        out = out.permute(1, 0, 2)
        out = torch.sum(out, 1)
        out = self.fc(out)
        return out


def prepare_fin(config):
    fin = open(config.result_file, 'a')
    fin.write('-------------------------------------\n')
    fin.write(config.model_name + '\n')
    fin.write(config.mode + '\n')
    fin.write('Begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + '\n')
    fin.write('Data root dir: ' + config.root_dir + '\n')
    fin.write('d_model: ' + str(config.d_model) + '\t pad_size: ' + str(config.pad_size) + '\t nhead: ' + str(config.nhead)
            + '\t num_layers: ' + str(config.num_layers) + '\n')
    fin.write(
        'batch_size: ' + str(config.batch_size) + '\t learning rate: ' + str(
            config.lr) + '\t smooth factor: ' + str(config.gran) + '\n\n')
    fin.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="")
    parser.add_argument('--window_size', type=int, default=15)
    parser.add_argument('--type', type=str, default="chd")
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mode', type=str, default="ae")
    parser.add_argument('--tse', type=bool, default=False)
    parser.add_argument('--ver', type=str, default='1')
    args = parser.parse_args()
    
    config = Config(args)
    prepare_fin(config)
    
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    train_dataset = DatasetPreprocess(config.root_dir, config.window_size, config.pad_size, config.d_model, config.max_time_position, config.gran, config.log_e, is_train=True)
    test_dataset = DatasetPreprocess(config.root_dir, config.window_size, config.pad_size, config.d_model, config.max_time_position, config.gran, config.log_e, is_train=False)

    print("TRAIN SIZE:", len(train_dataset), " TEST SIZE:", len(test_dataset), " SIZE:", len(train_dataset)+len(test_dataset), " TRAIN RATIO:", round(len(train_dataset)/(len(train_dataset)+len(test_dataset))*100), "%")
    # print("TRAIN DATA:", len(train_dataset[0]['header']))


    # 2 DataLoader
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
        model = TransformerPredictor(config)
        start_epoch = -1
    loss_func = nn.CrossEntropyLoss().to(config.device)
    opt = optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = CosineWarmupScheduler(opt, warmup=50, max_iters=config.epoch_num*len(train_loader))


    for epoch in range(start_epoch + 1, config.epoch_num):
        fin = open(config.result_file, 'a')
        print('--- epoch ', epoch)
        fin.write('-- epoch ' + str(epoch) + '\n')
        for i, sample_batch in enumerate(train_loader):
            batch_header = sample_batch['header'].type(torch.FloatTensor).to(config.device)
            batch_payload = sample_batch['payload'].type(torch.FloatTensor).to(config.device)
            batch_mask = sample_batch['mask'].to(config.device)
            batch_label = sample_batch['label'].to(config.device)
            batch_time_position = sample_batch['time'].to(config.device)
            
            out = model(batch_header, batch_payload, batch_mask, batch_time_position)
            loss = loss_func(out, batch_label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
            if i % 20 == 0:
                print('iter {} loss: '.format(i), loss.item())
        torch.save(model, (config.model_save_path + config.model_name + '_model_{}.pth').format(epoch))

        # test
        label_y = []
        pre_y = []
        with torch.no_grad():
            for j, test_sample_batch in enumerate(test_loader):
                test_header = test_sample_batch['header'].type(torch.FloatTensor).to(config.device)
                test_payload = test_sample_batch['payload'].type(torch.FloatTensor).to(config.device)
                test_mask = test_sample_batch['mask'].to(config.device)
                test_label = test_sample_batch['label'].to(config.device)
                test_time_position = test_sample_batch['time'].to(config.device)
                
                test_out = model(test_header, test_payload, test_mask, test_time_position)

                pre = torch.max(test_out, 1)[1].cpu().numpy()
                
                pre_y = np.concatenate([pre_y, pre], 0)
                label_y = np.concatenate([label_y, test_label.cpu().numpy()], 0)
            write_result(fin, label_y, pre_y, config.classes_num)
            draw_confusion(label_y, pre_y, '')
        fin.close()

    fin = open(config.result_file, 'a')
    fin.write('\n\n\n')
    fin.close()
    






