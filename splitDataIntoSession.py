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
import os
import helper_functions
from copy import copy, deepcopy

attributes = ['Timestamp', 'canID', 'DLC', 
                           'Data0', 'Data1', 'Data2', 
                           'Data3', 'Data4', 'Data5', 
                           'Data6', 'Data7', 'Flag']

road_attributes = ['Timestamp', 'canID', 'Data', 'TimeDiffs', 'Flag']

def hex_to_int(hex_value):
    return int(hex_value, base=16)

def hex_string_to_array(hex_string):
    return list(map(hex_to_int, hex_string))

def complete_field(sample):
    if not isinstance(sample['Flag'], str):
        col = 'Data' + str(sample['DLC'])
        sample['Flag'] = sample[col]
        sample[col] = '00'
    return sample

def split_into_list(string, type):
    # print(payload)
    res = []
    if type == 'payload':
        for i in range(7):
            res.append(string[:2])
            string = string[2:]
        res.append(string[-2:])
    else:
        hex_cid = '0' + hex(string)[2:] if len(hex(string)[2:]) == 3 else '00' + hex(string)[2:] if len(hex(string)[2:]) == 2 else '000' + hex(string)[2:]
        for i in range(3):
            res.append(hex_cid[:1])
            hex_cid = hex_cid[1:]
        res.append(hex_cid[-1:])
    return hex_string_to_array(res)


def serialize_example(x, y): 
    """converts x, y to tf.train.Example and serialize"""
    #Need to pay attention to whether it needs to be converted to numpy() form
    timestamp, canid, payload = x
    timestamp = tf.train.FloatList(value = np.array(timestamp).flatten())
    canid = tf.train.Int64List(value = np.array(canid).flatten())
    payload = tf.train.Int64List(value = np.array(payload).flatten())
    label = tf.train.Int64List(value = np.array([y]))
    features = tf.train.Features(
        feature = {
            "timestamp": tf.train.Feature(float_list = timestamp),
            "header": tf.train.Feature(int64_list = canid),
            "payload": tf.train.Feature(int64_list = payload),
            "label" : tf.train.Feature(int64_list = label)
        }
    )
    example = tf.train.Example(features = features)
    return example.SerializeToString()

def write_tfrecord(data, filename):
    tfrecord_writer = tf.io.TFRecordWriter(filename)
    for _, row in tqdm(data.iterrows()):
        X = (row['timestamp'], row['header'], row['payload'])
        Y = row['label']
        tfrecord_writer.write(serialize_example(X, Y))
    tfrecord_writer.close() 

def split_data(file_name, attack_id, window_size, strided_size, type_data):
    if not os.path.exists(file_name):
        print(file_name, ' does not exist!')
        return None
    
    print("Window size = {}, strided = {}".format(window_size, strided_size))
    
    if type_data == 'chd':
        df = pd.read_csv(file_name, header=None, names=attributes)
        print("Reading {}: done".format(file_name))
        df = df.sort_values('Timestamp', ascending=True)
        df = df.swifter.apply(complete_field, axis=1) 
        
        num_data_bytes = 8
        for x in range(num_data_bytes):
            df['Data'+str(x)] = df['Data'+str(x)].map(lambda x: int(x, 16), na_action='ignore')
            
        df['canID'] = df['canID'].apply(lambda x: hex_string_to_array(x))
        
        df = df.fillna(0)
        data_cols = ['Data{}'.format(x) for x in range(num_data_bytes)]
        df[data_cols] = df[data_cols].astype(int) 
        df['Data'] = df[data_cols].values.tolist()
        df['Flag'] = df['Flag'].apply(lambda x: True if x=='T' else False)
        print("Car hacking pre-processing: Done")
    else:
        df = pd.read_csv(file_name)
        df.columns = road_attributes
        print("Reading {}: done".format(file_name))
        df = df.sort_values('Timestamp', ascending=True)
        df['Data'] = df['Data'].apply(lambda x: split_into_list(x, 'payload'))
        print("Copy anomaly: DONE")
        df['canID'] = df['canID'].apply(lambda x: split_into_list(x, 'cid'))
        print("ROAD pre-processing: Done") 
    
    
    as_strided = np.lib.stride_tricks.as_strided
    output_shape = ((len(df) - window_size) // strided_size + 1, window_size)
    timestamp = as_strided(df.Timestamp, output_shape, (8*strided_size, 8))
    canid = as_strided(df.canID, output_shape, (8*strided_size, 8))
    data = as_strided(df.Data, output_shape, (8*strided_size, 8)) #Stride is counted by bytes
    label = as_strided(df.Flag, output_shape, (1*strided_size, 1))
    
    # print("timestamp output", timestamp[0], " and length: ", len(timestamp[0]))
    # print("canid output", canid[0], " and length: ", len(canid[0]))
    # print("data output", data[0], " and length: ", len(data[0]))
    # print("label output", label[0], " and length: ", len(label[0]))
    
    df = pd.DataFrame({
        'timestamp': pd.Series(timestamp.tolist()), 
        'header': pd.Series(canid.tolist()), 
        'payload': pd.Series(data.tolist()),
        'label': pd.Series(label.tolist())
    }, index= range(len(canid)))
    
    # print('Label before use: ', df['label'])
    df['label'] = df['label'].apply(lambda x: attack_id if any(x) else 0)
    print("Aggregating data: Done")
    print('#Normal: ', df[df['label'] == 0].shape[0])
    print('#Attack: ', df[df['label'] != 0].shape[0])
    return df[['timestamp', 'header', 'payload', 'label']].reset_index().drop(['index'], axis=1)

def main(indir, outdir, attacks, window_size, strided, attack_types):
    print(outdir)
    print("========================================================================================")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data_info = {} 
    
    if len(attacks) > 4:
        type_data = 'road'
        # process training data
        # normal_data = 'aggregated_training'
        # finput = '{}/{}_data.csv'.format(indir, normal_data)
        # df, df_anomaly = split_data(finput, 0, window_size, strided, type_data)
        # df_aggregation.append(df_anomaly)
        # print("Writing Normal...................")
        # foutput_normal = '{}/Normal_{}'.format(outdir, normal_data)
        # write_tfrecord(df, foutput_normal)
        # data_info[foutput_normal] = df.shape[0]
        
        # process attack data
        for attack_id, attack in enumerate(attacks):
            # Split to get number of dataset
            attack_name = attack.split(',')[0]
            attack_ver = attack.split(',')[1]
            print('Attack: {} ==============='.format(attack_name))
            
            if int(attack_ver) == 1:
                if attack_types == 'road_fab':
                    finput = '{}/{}_dataset.csv'.format(indir, attack_name)
                    df = split_data(finput, attack_id + 1, window_size, strided, type_data)
                elif attack_types == 'road_mas':
                    finput = '{}/{}_masquerade_dataset.csv'.format(indir, attack_name)
                    df = split_data(finput, attack_id + 1, window_size, strided, type_data)
                else:
                    df = []
                    finput = '{}/{}_dataset.csv'.format(indir, attack_name)
                    df_1 = split_data(finput, attack_id + 1, window_size, strided, type_data)
                    df.append(df_1)
                    finput = '{}/{}_masquerade_dataset.csv'.format(indir, attack_name)
                    df_2 = split_data(finput, attack_id + 1, window_size, strided, type_data)
                    df.append(df_2)
                    df = pd.concat(df)
                    
                print("Writing...................")
                foutput_attack = '{}/{}'.format(outdir, attack_name)
                foutput_normal = '{}/Normal_{}'.format(outdir, attack_name)
                df_attack = df[df['label'] != 0]
                df_normal = df[df['label'] == 0]
                write_tfrecord(df_attack, foutput_attack)
                write_tfrecord(df_normal, foutput_normal)
                data_info[foutput_attack] = df_attack.shape[0]
                data_info[foutput_normal] = df_normal.shape[0]
            else:
                for index in range(int(attack_ver)):
                    if attack_types == 'road_fab':
                        finput = '{}/{}_{}_dataset.csv'.format(indir, attack_name, index+1)
                        df = split_data(finput, attack_id + 1, window_size, strided, type_data)
                    elif attack_types == 'road_mas':
                        finput = '{}/{}_{}_masquerade_dataset.csv'.format(indir, attack_name, index+1)
                        df = split_data(finput, attack_id + 1, window_size, strided, type_data)
                    else:
                        df = []
                        finput = '{}/{}_{}_dataset.csv'.format(indir, attack_name, index+1)
                        df_1 = split_data(finput, attack_id + 1, window_size, strided, type_data)
                        df.append(df_1)
                        finput = '{}/{}_{}_masquerade_dataset.csv'.format(indir, attack_name, index+1)
                        df_2 = split_data(finput, attack_id + 1, window_size, strided, type_data)
                        df.append(df_2)
                        df = pd.concat(df) 
                        
                    print("Writing...................")
                    foutput_attack = '{}/{}_{}'.format(outdir, attack_name, index+1)
                    foutput_normal = '{}/Normal_{}_{}'.format(outdir, attack_name, index+1)
                    df_attack = df[df['label'] != 0]
                    df_normal = df[df['label'] == 0]
                    write_tfrecord(df_attack, foutput_attack)
                    write_tfrecord(df_normal, foutput_normal)
                    
                    data_info[foutput_attack] = df_attack.shape[0]
                    data_info[foutput_normal] = df_normal.shape[0]
    elif len(attacks) == 4: 
        type_data = 'chd'
        for attack_id, attack in enumerate(attacks):
            print('Attack: {} ==============='.format(attack))
            finput = '{}/{}_dataset.csv'.format(indir, attack)
            df = split_data(finput, attack_id + 1, window_size, strided, type_data)
            print("Writing...................")
            foutput_attack = '{}/{}'.format(outdir, attack)
            foutput_normal = '{}/Normal_{}'.format(outdir, attack)
            df_attack = df[df['label'] != 0]
            df_normal = df[df['label'] == 0]
            write_tfrecord(df_attack, foutput_attack)
            write_tfrecord(df_normal, foutput_normal)
            
            data_info[foutput_attack] = df_attack.shape[0]
            data_info[foutput_normal] = df_normal.shape[0]
    print("Write record DONE!!!")
    json.dump(data_info, open('{}/datainfo.txt'.format(outdir), 'w'))
    print("DONE!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="./data/car-hacking")
    parser.add_argument('--outdir', type=str, default="./data/CHD/TFRecord")
    parser.add_argument('--window_size', type=int, default=15)
    parser.add_argument('--strided', type=int, default=15)
    parser.add_argument('--attack_type', type=str, default="all", nargs='+')
    args = parser.parse_args()
    
    if args.attack_type[0] == 'chd':
        attack_types = ['DoS', 'Fuzzy', 'gear', 'RPM']
        print("Car hacking")
    elif args.attack_type[0] == 'road_fab':
        attack_types = ['max_engine_coolant_temp_attack,1', 'fuzzing_attack,3', 'max_speedometer_attack,3', 'reverse_light_on_attack,3', 'reverse_light_off_attack,3', 'correlated_signal_attack,3']
    elif args.attack_type[0] == 'road_mas':
        attack_types = ['max_engine_coolant_temp_attack,1', 'max_speedometer_attack,3', 'reverse_light_on_attack,3', 'reverse_light_off_attack,3', 'correlated_signal_attack,3']
    elif args.attack_type[0] == 'all':
        attack_types = ['max_engine_coolant_temp_attack,1', 'fuzzing_attack,3', 'max_speedometer_attack,3', 'reverse_light_on_attack,3', 'reverse_light_off_attack,3', 'correlated_signal_attack,3']
    else:
        attack_types = [args.attack_type]
    
    if args.strided == None:
        args.strided = args.window_size
        
    outdir =  args.outdir + '_w{}_s{}'.format(args.window_size, args.strided)
    main(args.indir, outdir, attack_types, args.window_size, args.strided, args.attack_type[0])