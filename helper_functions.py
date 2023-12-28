import json
import pickle
import codecs
import copy
import datetime
import os

import numpy as np
import os

import matplotlib.pyplot as plt
#%matplotlib inline

import pandas as pd

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import scipy.stats
#from generalFunctions import *
import cProfile
from collections import Counter
import bisect

import re
from tqdm import tqdm

from sklearn.covariance import EllipticEnvelope


# functions for saving/opening objects
def jsonify(obj, out_file):
    """
    Inputs:
    - obj: the object to be jsonified
    - out_file: the file path where obj will be saved
    This function saves obj to the path out_file as a json file.
    """
    json.dump(obj, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def unjsonify(in_file):
    """
    Input:
    -in_file: the file path where the object you want to read in is stored
    Output:
    -obj: the object you want to read in
    """
    obj_text = codecs.open(in_file, 'r', encoding='utf-8').read()
    obj = json.loads(obj_text)
    return obj

def picklify(obj, filepath):
    """
    Inputs:
    - obj: the object to be pickled
    - filepath: the file path where obj will be saved
    This function pickles obj to the path filepath.
    """
    pickle_file = open(filepath, "wb")
    pickle.dump(obj, pickle_file)
    pickle_file.close()
    #print "picklify done"


def unpickle(filepath):
    """
    Input:
    -filepath: the file path where the pickled object you want to read in is stored
    Output:
    -obj: the object you want to read in
    """
    pickle_file = open(filepath, 'rb')
    obj = pickle.load(pickle_file)
    pickle_file.close()
    return obj

def curtime_str():
    """A string representation of the current time."""
    dt = datetime.datetime.now().time()
    return dt.strftime("%H:%M:%S")


def update_json_dict(key, value, out_file, overwrite = True):
    if not os.path.isfile(out_file):
        d = {}
    else:
        d = unjsonify(out_file)
        if key in d and not overwrite:
            print("fkey {key} already in {out_file}, skipping...")
            return
    d[key] = value
    jsonify(d, out_file)

    #jsonify(sorted(d.items(), key = lambda x: x[0]), out_file)


def make_can_df(log_filepath):
    """
    Puts candump data into a dataframe with columns 'time', 'aid', and 'data'
    """
    can_df = pd.read_fwf(
        log_filepath, delimiter = ' '+ '#' + '('+')',
        skiprows = 1,skipfooter=1,
        usecols = [0,2,3],
        dtype = {0:'float64', 1:str, 2: str},
        names = ['time','aid', 'data'] )

    can_df.aid = can_df.aid.apply(lambda x: int(x,16))
    can_df.data = can_df.data.apply(lambda x: x.zfill(16)) #pad with 0s on the left for data with dlc < 8
    can_df.time = can_df.time - can_df.time.min()
    
    return can_df[can_df.aid<=0x700] # out-of-scope aid


def add_time_diff_per_aid_col(df, order_by_time = False):
    """
    Sorts df by aid and time and takes time diff between each successive col and puts in col "time_diffs"
    Then removes first instance of each aids message
    Returns sorted df with new column
    """

    df.sort_values(['aid','time'], inplace=True)
    df['time_diffs'] = df['time'].diff()
    mask = df.aid == df.aid.shift(1) #get bool mask of to filter out first msg of each group
    df = df[mask]
    if order_by_time:
        df = df.sort_values('time').reset_index()
    return df


def get_injection_interval(df, injection_aid, injection_data_str, max_injection_t_delta=1):
    """
    Compute time intervals where attacks were injected based on aid and payload
    @param df: testing df to be analyzed (dataframe)
    @param injection_aid: aid that injects the attack (int)
    @param injection_data_str: payload of the attack (str)
    @param max_injection_t_delta: minimum separation between attacks (int)
    @output injection_intervals: list of intervals where the attacks were injected (list)
    """
    
    # Construct a regular expression to identify the payload
    injection_data_str = injection_data_str.replace("X", ".")

    attack_messages_df = df[(df.aid==injection_aid) & (df.data.str.contains(injection_data_str))] # get subset of attack messages
    #print(attack_messages_df)

    if len(attack_messages_df) == 0:
        print("message not found")
        return None

    # Assuming that attacks are injected with a diferrence more than i seconds
    inj_period_times = np.split(np.array(attack_messages_df.time),
        np.where(attack_messages_df.time.diff()>max_injection_t_delta)[0])

    # Pack the intervals
    injection_intervals = [(time_arr[0], time_arr[-1])
        for time_arr in inj_period_times if len(time_arr)>1]

    return injection_intervals


def add_actual_attack_col(df, intervals, aid, payload, attack_name):
    """
    Adds column to df to indicate which signals were part of attack
    """

    if aid != "XXX":
        if attack_name.startswith('correlated_signal'):
            df['label'] = df.time.apply(lambda x: sum(x >= intvl[0]  and x <= intvl[1] for intvl in intervals ) >= 1) & (df.aid == aid) & (df.data == payload)
        elif attack_name.startswith('max'):
            df['label'] = df.time.apply(lambda x: sum(x >= intvl[0]  and x <= intvl[1] for intvl in intervals ) >= 1) & (df.aid == aid) & df.data.str.contains(payload[10:12], regex=False)
        else:
            df['label'] = df.time.apply(lambda x: sum(x >= intvl[0]  and x <= intvl[1] for intvl in intervals ) >= 1) & (df.aid == aid) & df.data.str.contains(payload[4:6], regex=False)
    else:
        df['label'] = df.time.apply(lambda x: sum(x >= intvl[0]  and x <= intvl[1] for intvl in intervals ) >= 1) & (df.data == payload)
    return df
