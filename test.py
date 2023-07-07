import numpy as np
import torch
import os
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import Dataset

ID_LEN = 29 #CAN bus 2.0 has 29 bits
DATA_LEN = 8 #Data field in Can message has 8 bytes
HIST_LEN = 256

class CANDataset(Dataset):
    def __init__(self, root_dir, window_size, is_train=True, include_data=False, transform=None):
        if is_train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')
            
        # self.num_classes = num_classes
        self.include_data = include_data
        self.is_train = is_train
        self.transform = transform
        self.window_size = window_size
        self.total_size = len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        filenames = '{}/{}.tfrec'.format(self.root_dir, idx)
        index_path = None
        description = {'id_seq': 'int', 'data_seq': 'int','label': 'int'}
        dataset = TFRecordDataset(filenames, index_path, description)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        data = next(iter(dataloader))
        id_seq, data_seq, label = data['id_seq'], data['data_seq'], data['label']
        id_seq = id_seq.to(torch.float)
        data_seq = data_seq.to(torch.float)
        
        id_seq[id_seq == 0] = -1
        id_seq = id_seq.view(-1, self.window_size, ID_LEN)
        data_seq = data_seq.view(-1, self.window_size, DATA_LEN)
        
        if self.include_data:
            return id_seq, data_seq, label[0][0]
        else:
            return id_seq, label[0][0]
        
    def __len__(self):
        return self.total_size