import numpy as np
import torch
import os
from tfrecord.torch.dataset import TFRecordDataset
import math
import torch.nn.functional as F
from torch.utils.data import Dataset

ID_LEN = 4 #CAN bus 2.0 has 29 bits
DATA_LEN = 8 #Data field in Can message has 8 bytes

class DatasetPreprocess(Dataset):
    def __init__(self, root_dir, window_size, pad_size, embed, max_time_position, gran, log_e, transform=None, is_train=True):
        if is_train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')
        self.transform = transform
        self.pad_size = pad_size
        self.embed = embed
        self.max_time_position = max_time_position
        self.gran = gran
        self.log_e = log_e
        self.window_size = window_size
        self.total_size = len(os.listdir(self.root_dir))

        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / self.embed)) for i in range(self.embed)] for pos in
             range(self.max_time_position)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # Use sin for even columns
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  # Use cos for odd columns
        
    def __len__(self):
        return self.total_size
    
    def get_time(self, time_position):
        # Segment the corresponding position code according to the time position
        pe = torch.index_select(self.pe, 0, time_position)
        return pe
    
    def __getitem__(self, idx):
        filenames = '{}/{}.tfrec'.format(self.root_dir, idx)
        print("File name: ", filenames)
        
        if not os.path.isfile(filenames):
            print(filenames + 'does not exist!')
            
        index_path = None
        description = {'timestamp' : 'float','header': 'int', 'payload': 'int','label': 'int'}
        
        dataset = TFRecordDataset(filenames, index_path, description)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        data = next(iter(dataloader))
        
        timestamp, header, payload, label = data['timestamp'], data['header'], data['payload'], data['label']
        
        # timestamp = timestamp.to(torch.float)
        # print("TIMESTAMP: ", timestamp, "AND LENGTH: ", len(timestamp[0]))
        # print("HEADER: ", (header), "AND LENGTH: ", len(header[0]))
        # print("PAYLOAD: ", type(payload), "AND LENGTH: ", len(payload[0]))
        # print("LABEL: ", type(label), "AND LENGTH: ", len(label[0]))
        
        # header[header == 0] = -1
        timestamp = timestamp.numpy()[0]
        header = np.reshape(header.numpy(), (self.window_size, ID_LEN))
        payload = np.reshape(payload.numpy(), (self.window_size, DATA_LEN))
        label = label.numpy()[0][0]
        
        # print("TIMESTAMP AFTER: ", (timestamp), "AND LENGTH: ", len(timestamp))
        # print("HEADER AFTER: ", header, "AND LENGTH: ", type(header))
        # print("PAYLOAD AFTER: ", (payload), "AND LENGTH: ", len(payload))
        # print("LABEL AFTER: ", (label))
        # print("DONE")

        # # PREPROCESS
        # ## GET HEADER 
        # header = np.array(list(map(hex_string_to_array, list(false_data['canID']))))[:100]
        header = torch.from_numpy(header)
        # # print("HEADER BEFORE: ", header)
        # ## GET PAYLOAD 
        # payload = np.array(list(map(hex_string_to_array, list(false_data['Payload']))))
        payload = torch.from_numpy(payload)
        # # print("PAYLOAD BEFORE: ",payload)

        ori_seq_len = header.shape[0]
        pad_len = 100 - ori_seq_len
        print(pad_len)
        ## PAD WITH MAX SIZE = 100
        header = F.pad(header.T, (0, pad_len)).T.numpy()
        payload = F.pad(payload.T, (0, pad_len)).T.numpy()

        if pad_len == 0:
            mask = np.array([False] * ori_seq_len)
        else:
            mask = np.concatenate((np.array([False] * ori_seq_len), np.array([True] * pad_len)))
        
        ## GET TIMESTAMP
        len_timestamp = len(timestamp)

        for i in range(len_timestamp):
            value = round(math.log(round(timestamp[i] / self.gran) + 1, self.log_e))
            timestamp[i] = value
        for j in range(self.pad_size - len_timestamp):
            timestamp = np.append(timestamp, timestamp[len_timestamp - 1])

        time_feature = self.get_time(torch.IntTensor(timestamp))
        # label = false_data['Flag']
        sample = {'header': header, 'payload': payload, 'mask': mask, 'time': time_feature, 'label': label, 'idx': idx}
        
        print("HEADER FEATURE: ", header, " AND LENGTH: ", len(header))
        print("PAYLOAD FEATURE: ", payload, " AND LENGTH: ", len(payload[0]))
        print("MASK FEATURE: ", mask, " AND LENGTH: ", len(mask))
        print("TIME FEATURE: ", time_feature)
        print("LABEL: ", label)
        print("INDEX: ", idx)
        print("DONE")
        
        if self.transform:
            sample = self.transform(sample)

        # # print("SAMPLE: ", sample)
        return sample

