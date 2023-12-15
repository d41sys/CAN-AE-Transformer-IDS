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

    #print(can_df)
    # can_df['label'] = ['T']*can_df.shape[0]
    can_df.aid = can_df.aid.apply(lambda x: int(x,16))
    can_df.data = can_df.data.apply(lambda x: x.zfill(16)) #pad with 0s on the left for data with dlc < 8
    can_df['ori_time'] = can_df.time
    can_df.time = can_df.time - can_df.time.min()

    #print(can_df)
    return can_df[can_df.aid<=0x700]

def extract_k_bits(num, p=8, k=18):
    """
    Extract decimal representation of a hex string
    @param num: Hex to be analyzed (str)
    @param p: Start of the slice (int). It is 9 for J1939
    @param k: Length of the slide (int). It is 18 for J1939
    @output : Decimal representation of the slice (int)
    """

    # Convert hex str into a 29 bit representation (J1939 AID)
    binary = format(int(num, 16), "029b")
    #print(binary)

    # Compute slice indices
    end = len(binary) - p
    start = end - k + 1

    # Slice the binary string
    k_bit_string = binary[start:end+1]

    # Convert binary string to decimal
    value = int(k_bit_string, 2)
    #print(value)

    return(value)

def make_1939_df(log_filepath):
    """
    Puts candump data into a dataframe with columns 'time', 'PGN', and 'data'
    @param log_filepath: Path to the file (str)
    @output can_df: Data frame representation of the log file (data frame)
    """
    can_df = pd.read_fwf(
        log_filepath, delimiter=" " + "#" + "(" + ")",
        skiprows=0, skipfooter=0,
        usecols=[0, 2, 3],
        dtype={0:"float64", 1:str, 2:str},
        names=["time", "aid", "data"] )

    # Discard rows with any NaN
    can_df = can_df[can_df["aid"].notna()]

    # print(can_df.dtypes)
    # display(can_df)

    # can_df.aid = can_df.aid.apply(lambda x: int(x,16))

    can_df.aid = can_df.aid.apply(extract_k_bits)
    can_df.data = can_df.data.apply(lambda x: x.zfill(16)) #pad with 0s on the left for data with dlc < 8
    can_df.time = can_df.time - can_df.time.min()

    can_df.columns = ["time", "PGN", "data"]

    return can_df

def extract_numbers(string_to_parse):
    """
    Extract the first string number (either int of float) from a string
    @param string_to_parse: Input string (str)
    @output res: number representation (float)
    """

    temp = re.search(r"\d+\.*\d+", string_to_parse)

    if temp:
        res = float(temp.group())
        # Transform to seconds
        if res > 1:
            res = res/1000
    else:
        res = ""

    return res


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


def add_time_diff_per_PGN_col(df, order_by_time = False):
    """
    Sorts df by aid and time and takes time diff between each successive col and puts in col "time_diffs"
    Then removes first instance of each aids message
    Returns sorted df with new column
    """

    df.sort_values(['PGN','time'], inplace=True)
    df['time_diffs'] = df['time'].diff()
    mask = df.PGN == df.PGN.shift(1) #get bool mask of to filter out first msg of each group
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


def add_kde_val_col(df, d):
    """
    Adds column to df with the value of the kde at each time_diff in the df
    """
    # df['kde_val'] = df.apply(lambda row: d[row.aid]['kde'].evaluate(row.time_diffs)[0], axis=1)
    new_column = np.concatenate([d[aid]["kde"].evaluate(df[df.aid == aid].time_diffs.values) for aid in tqdm(df.aid.unique())])
    df['kde_val'] = new_column

    return df


def add_gauss_val_col(df, d):
    """
    Adds column to df with the value of the Guassian approximation at each time_diff in the df
    """
    # df['gauss_val'] = df.apply(lambda row: d[row.aid]['gauss'].pdf(row.time_diffs), axis = 1)
    new_column = np.concatenate([d[aid]["gauss"].pdf(df[df.aid == aid].time_diffs.values) for aid in tqdm(df.aid.unique())])
    df['gauss_val'] = new_column
    return df


def train(df, aid):
    """
    Returns a dictionary the aid including the mean of its time_diffs, standard deviation of its time_diffs
    and KDE of its time_diffs
    """
    time_diffs = df[df.aid==aid].time_diffs.values
    print("before: ", len(time_diffs))

    # identify outliers in the dataset
    ee = EllipticEnvelope(contamination=0.0001, support_fraction=0.999) # support_fraction=0.99
    inliers = ee.fit_predict(time_diffs.reshape(-1, 1))

    # select all rows that are not outliers
    mask = inliers != -1
    outliers = sum(mask == False)
    print("outliers: ", outliers, 100*outliers/len(time_diffs))

    time_diffs = time_diffs[mask]
    # summarize the shape of the updated training dataset
    print("after: ", len(time_diffs))

    aid_dict = {'mu': time_diffs.mean(), 'std': time_diffs.std(), 'kde': scipy.stats.gaussian_kde(time_diffs), 'gauss': scipy.stats.norm(loc = time_diffs.mean(), scale = time_diffs.std())}
    return aid_dict

def train_no_preprocessing(df, aid):
    """
    Returns a dictionary the aid including the mean of its time_diffs, standard deviation of its time_diffs
    and KDE of its time_diffs
    """
    time_diffs = df[df.aid==aid].time_diffs.values
    # print("before: ", len(time_diffs))

    # # identify outliers in the dataset
    # ee = EllipticEnvelope(contamination=0.0001, support_fraction=0.999) # support_fraction=0.99
    # inliers = ee.fit_predict(time_diffs.reshape(-1, 1))

    # # select all rows that are not outliers
    # mask = inliers != -1
    # outliers = sum(mask == False)
    # print("outliers: ", outliers, 100*outliers/len(time_diffs))

    # time_diffs = time_diffs[mask]
    # # summarize the shape of the updated training dataset
    # print("after: ", len(time_diffs))

    aid_dict = {'mu': time_diffs.mean(), 'std': time_diffs.std(), 'kde': scipy.stats.gaussian_kde(time_diffs), 'gauss': scipy.stats.norm(loc = time_diffs.mean(), scale = time_diffs.std())}
    return aid_dict


def train_J1939(df, PGN):
    """
    Returns a dictionary the aid including the mean of its time_diffs, standard deviation of its time_diffs
    and KDE of its time_diffs
    """
    time_diffs = df[df["PGN"] == PGN].time_diffs
    aid_dict = {'mu': time_diffs.mean(), 'std': time_diffs.std()}
    return aid_dict


def y_threshold_kde(dict, aid, p):
    """
    Determines the approximate y value at which the KDE of the aid has the desired p value
    """
    pvs = []
    mu = dict[aid]['mu']
    std = dict[aid]['std']
    x = np.arange(mu - 5*std, mu + 5*std, 10*std/1000)
    #y = [dict[aid]['kde'].evaluate(i_x) for i_x in x]
    y = dict[aid]['kde'].evaluate(x)
    for i, i_x in enumerate(x):
        pvs.append(sum([j for j in y if j <= y[i] ]) * 10*std/1000)
    if np.where(np.array(pvs) <= p)[0] != []:
        y_threshold = np.max(np.array(y)[np.where(np.array(pvs) <= p)[0]])
    else: y_threshold = 0
    dict[aid]['y_thresholds_kde'][p] = y_threshold


def y_threshold_gauss(dict, aid, p):
    """
    Determines the approximate y value at which the Gaussian approximation of the aid has the desired p value
    """

    pvs = []
    mu = dict[aid]['mu']
    std = dict[aid]['std']
    x = np.arange(mu - 5*std, mu + 5*std, 10*std/1000)
    # y = [dict[aid]['gauss'].pdf(i_x) for i_x in x]
    y = dict[aid]['gauss'].pdf(x)
    for i, i_x in enumerate(x):
        pvs.append(sum([j for j in y if j <= y[i] ]) * 10*std/1000)
    y_threshold = np.max(np.array(y)[np.where(np.array(pvs) <= p)[0]])
    dict[aid]['y_thresholds_gauss'][p] = y_threshold


def alert_by_mean(df, d):
    """
    Adds column to df to indicate when time_diff is less than half the mean time_diff for the aid
    """
    df['predicted_attack'] = df.apply(lambda row: row.time_diffs <= (d[row.aid]['mu']/2), axis = 1)
    return df


def alert_by_mean_various_p(df, d, p):
    """
    Adds column to df to indicate when time_diff is less than half the mean time_diff for the aid
    """
    df['predicted_attack'] = df.apply(lambda row: row.time_diffs <= (p*d[row.aid]['mu']), axis=1)
    return df


def alert_by_mean_2(df, dict,n):
    """
    Same as alert_by_mean but slower
    """
    predicted_attack = []
    for i in range(df.count()[0]):
        aid = df.iloc[i].aid
        time_diff = df.iloc[i].time_diffs
        if aid in dict.keys():
            if time_diff <= (dict[aid]['mu'])/n:
                predicted_attack.append(True)
            else:
                predicted_attack.append(False)
        else:
            #print("aid %s not seen in training; no attacks predicted" %(aid))
            predicted_attack.append(False)
    df['predicted_attack'] = predicted_attack
    return df

#df_attack['kde_val'] = df_attack.apply(lambda row: d[row.aid]['kde'].evaluate(row.time_diffs)[0], axis = 1)

def alert_by_kde(df,d, p):
    """
    Adds column to df that labels at which time_diffs the value of kde is less than the desired kde threshold
    """
    df['predicted_attack'] = df.apply(lambda row: row.kde_val <= d[row.aid]['y_thresholds_kde'][p], axis = 1)
    return df


def alert_by_kde_2(df, dict, p):
    """
    Same as alert_by_kde but much slower
    """
    predicted_attack = []
    for i in range(df.count()[0]):
        aid = df.iloc[i].aid
        time_diff = df.iloc[i].time_diffs
        if dict[aid]['kde'].evaluate(time_diff) <= dict[aid]['y_thresholds_kde'][p]:
            predicted_attack.append(True)
        else:
            predicted_attack.append(False)
    df['predicted_attack'] = predicted_attack
    return predicted_attack


def alert_by_gauss(df, d, p):
    """
    Adds column to df that labels at which time_diffs the value of Gaussian estimate is less than the desired Gaussian threshold
    """
    df['predicted_attack'] = df.apply(lambda row: row.gauss_val <= d[row.aid]['y_thresholds_gauss'][p], axis = 1)
    return df


def alert_by_gauss_2(df, dict, p):
    """
    Same as alert_by_gauss but much slower
    """
    predicted_attack = []
    for i in range(df.count()[0]):
        aid = df.iloc[i].aid
        time_diff = df.iloc[i].time_diffs
        if dict[aid]['gauss'].pdf(time_diff) <= dict[aid]['y_thresholds_gauss'][p]:
            predicted_attack.append(True)
        else:
            predicted_attack.append(False)
    df['predicted_attack'] = predicted_attack
    return df


def signal_count_new(df, aid):
    """
    Calculates how many times a signal with this aid is seen in time window of length mu*4
    """
    
    # Get the mean inter-arrival time of messages for a specific aid
    # times_vec = np.array(df.time[df.aid==aid].to_list())
    times_vec = np.array(df.time[df.PGN==aid].to_list())
    times_vec = times_vec- times_vec.min()
    mu = np.diff(times_vec).mean()

    try:
        # Get the time breaking points in terms of positions
        breakpoints = [bisect.bisect_right(times_vec, i*4*mu) for i in range(int(max(times_vec)/(4*mu)))]

        diffs = list(np.diff(breakpoints))
        for i in range(len(diffs)):
            if diffs[i] > 9:
                diffs[i] = 9

        # Reflects the distribution od idstance between interarrival times. The limit is 9        
        count = Counter(np.concatenate([np.arange(0,10), np.array(diffs)]))
        return count
    except:
        return None


def prob_dict(df, aid):
    """
    Calculates the probability P[i] of seeing a signal with this aid i times in window of length mu*4
    Based on calculus and Bayes rules
    """
    count_aid = signal_count_new(df, aid)
    denom = sum(count_aid[i] for i in range(10))
    P = [0]*10
    for i in range(10):
        P[i] = count_aid[i]/denom
    return P


def alert_by_bin(df, d, n=6):
    """
    Checks for time windows of length mu*4 (where mu is average time_diff for aid) with 6 or more signals
    """
    pd.options.mode.chained_assignment = None

    cm = np.array([[0,0], [0,0]])
    for aid in df.aid.unique():
        #if d[aid]['std'] <= 0.01:
        df_test = df[df.aid==aid]
        df_test['predicted_attack'] = df_test.time_diffs.rolling(n).sum() <= d[aid]['mu']*4
        cm_aid = confusion_matrix(df_test['actual_attack'], df_test['predicted_attack'], labels = [0,1])
        cm += cm_aid
        #print(aid, cm_aid)
    return cm

def alert_by_bin_various_p(df, d, p, n=6):
    """
    Checks for time windows of length mu*4 (where mu is average time_diff for aid) with 6 or more signals
    """
    pd.options.mode.chained_assignment = None

    cm = np.array([[0,0], [0,0]])
    for aid in df.aid.unique():
        #if d[aid]['std'] <= 0.01:
        df_test = df[df.aid==aid]
        df_test['predicted_attack'] = df_test.time_diffs.rolling(n).sum() <= p*d[aid]['mu']
        cm_aid = confusion_matrix(df_test['actual_attack'], df_test['predicted_attack'], labels = [0,1])
        cm += cm_aid
        #print(aid, cm_aid)
    return cm


def alert_by_bin_J1939(df, d, n=6):
    """
    Checks for time windows of length mu*4 (where mu is average time_diff for aid) with 6 or more signals
    """
    cm = np.array([[0,0], [0,0]])
    for pgn in df["PGN"].unique():
        try :
            if d[pgn]['std'] <= 0.01:
                df_test = df[df["PGN"] == pgn]
                df_test['predicted_attack'] = df_test.time_diffs.rolling(n).sum() <= d[pgn]['mu']*4
                cm_aid = confusion_matrix(df_test['actual_attack'], df_test['predicted_attack'], labels = [0,1])
                cm += cm_aid
                #print(aid, cm_aid)
        except:
            pass
    return cm


def get_results_mean(attack_list, d):
    """
    Marks as attack when 3 or more out of last 6 time_diffs are less than half the mean
    Calulates confusion matrix, precision, recall, false positive rate and saves to dictionary
    """
    ## Initialize dictionary for results
    results_mean = {}
    for i in range(len(attack_list)):
        results_mean[i+1] = {'cm': [0], 'recall': 0, 'prec':0, 'false_pos':0}
    results_mean['total'] = {'cm': [0], 'recall': 0, 'prec':0, 'false_pos':0}

    for i in tqdm(range(len(attack_list))):
        attack_list[i] = alert_by_mean(attack_list[i], d)
        attack_list[i]['alert_window'] = attack_list[i].predicted_attack.rolling(6).sum() >= 3
        cm = confusion_matrix(attack_list[i]['actual_attack'], attack_list[i]['alert_window'])
        results_mean[i+1]['cm'] = cm
        results_mean[i+1]['prec'] = cm[1,1]/(cm[1,1]+cm[0,1])
        results_mean[i+1]['recall'] = cm[1,1]/(cm[1,1]+cm[1,0])
        results_mean[i+1]['false_pos'] = cm[0,1]/(cm[0,1] + cm[0,0])
        results_mean['total']['cm'] += cm
    results_mean['total']['prec'] =  results_mean['total']['cm'][1,1]/(results_mean['total']['cm'][1,1]+results_mean['total']['cm'][0,1])
    results_mean['total']['recall'] = results_mean['total']['cm'][1,1]/(results_mean['total']['cm'][1,1]+results_mean['total']['cm'][1,0])
    results_mean['total']['f1'] = 2*((results_mean['total']['prec']*results_mean['total']['recall'])/(results_mean['total']['prec']+results_mean['total']['recall']))
    results_mean['total']['false_pos'] = results_mean['total']['cm'][0,1]/(results_mean['total']['cm'][0,1]+results_mean['total']['cm'][0,0])

    # print(os.path.dirname(os.getcwd()))

    # return(results_mean)

    picklify(results_mean, os.path.dirname(os.getcwd()) + "/results_mean_final.pkl")


def get_results_mean_various_p(attack_list, d):
    """
    Marks as attack when 3 or more out of last 6 time_diffs are less than half the mean
    Calulates confusion matrix, precision, recall, false positive rate and saves to dictionary
    """

    pvals_mean = np.linspace(0, 1, 19)

    results_mean_final = {}
    for p in pvals_mean:
        results_mean_final[p] = {'cm': [0], 'recall': 0, 'prec': 0, 'false_pos': 0}

    for p in tqdm(pvals_mean):
        details = np.array([[0,0], [0,0]])
        for i in range(len(attack_list)):
            attack_list[i] = alert_by_mean_various_p(attack_list[i], d, p)
            attack_list[i]['alert_window'] = attack_list[i].predicted_attack.rolling(6).sum() >= 3
            details += confusion_matrix(attack_list[i]['actual_attack'], attack_list[i]['alert_window'])
        results_mean_final[p]['cm'] = details
        results_mean_final[p]['prec'] = details[1,1]/(details[1,1]+details[0,1])
        results_mean_final[p]['recall'] = details[1,1]/(details[1,1]+details[1,0])
        results_mean_final[p]['f1'] = 2*((results_mean_final[p]['prec']*results_mean_final[p]['recall'])/(results_mean_final[p]['prec']+results_mean_final[p]['recall']))
        results_mean_final[p]['false_pos'] = details[0,1]/(details[0,1] + details[0,0])

    # print(os.path.dirname(os.getcwd()))

    # return(results_mean_final)

    picklify(results_mean_final, os.path.dirname(os.getcwd()) + "/results_mean_final.pkl")


def get_results_mean_various_p_no_pretraining(attack_list, d):
    """
    Marks as attack when 3 or more out of last 6 time_diffs are less than half the mean
    Calulates confusion matrix, precision, recall, false positive rate and saves to dictionary
    """

    pvals_mean = np.linspace(0, 1, 19)

    results_mean_final = {}
    for p in pvals_mean:
        results_mean_final[p] = {'cm': [0], 'recall': 0, 'prec': 0, 'false_pos': 0}

    for p in tqdm(pvals_mean):
        details = np.array([[0,0], [0,0]])
        for i in range(len(attack_list)):
            attack_list[i] = alert_by_mean_various_p(attack_list[i], d, p)
            attack_list[i]['alert_window'] = attack_list[i].predicted_attack.rolling(6).sum() >= 3
            details += confusion_matrix(attack_list[i]['actual_attack'], attack_list[i]['alert_window'])
        results_mean_final[p]['cm'] = details
        results_mean_final[p]['prec'] = details[1,1]/(details[1,1]+details[0,1])
        results_mean_final[p]['recall'] = details[1,1]/(details[1,1]+details[1,0])
        results_mean_final[p]['f1'] = 2*((results_mean_final[p]['prec']*results_mean_final[p]['recall'])/(results_mean_final[p]['prec']+results_mean_final[p]['recall']))
        results_mean_final[p]['false_pos'] = details[0,1]/(details[0,1] + details[0,0])

    # print(os.path.dirname(os.getcwd()))

    # return(results_mean_final)

    picklify(results_mean_final, os.path.dirname(os.getcwd()) + "/results_mean_final_no_pretraining.pkl")


def get_results_kde(pvals, attack_list, d):
    """
    Marks as attack when last three time_diffs had had p-value less than the p-value threshold for kde
    Calculates confusion matrix, precision, recall, false positive rate and saves to dictionary
    """

    pvals_kde = sorted(list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0, 0.1, 0.01)))

    results_kde_final = {}
    for p in pvals_kde:
        results_kde_final[p] = {'cm': [0], 'recall': 0, 'prec': 0, 'false_pos': 0}

    for p in tqdm(pvals):
        details = np.array([[0,0], [0,0]])
        for i in range(len(attack_list)):
            attack_list[i] = alert_by_kde(attack_list[i], d, p)
            attack_list[i]['alert_window'] = attack_list[i].predicted_attack.rolling(3).sum() == 3
            details += confusion_matrix(attack_list[i]['actual_attack'], attack_list[i]['alert_window'])
        results_kde_final[p]['cm'] = details
        results_kde_final[p]['recall'] = details[1,1]/(details[1,1]+details[1,0])
        results_kde_final[p]['prec'] = details[1,1]/(details[1,1]+details[0,1])
        results_kde_final[p]['f1'] = 2*((results_kde_final[p]['prec']*results_kde_final[p]['recall'])/(results_kde_final[p]['prec']+results_kde_final[p]['recall']))
        results_kde_final[p]['false_pos'] = details[0,1]/(details[0,1]+details[0,0])
    picklify(results_kde_final, os.path.dirname(os.getcwd()) + "/results_kde_final.pkl")


def get_results_kde_no_pretraining(pvals, attack_list, d):
    """
    Marks as attack when last three time_diffs had had p-value less than the p-value threshold for kde
    Calculates confusion matrix, precision, recall, false positive rate and saves to dictionary
    """

    pvals_kde = sorted(list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0, 0.1, 0.01)))

    results_kde_final = {}
    for p in pvals_kde:
        results_kde_final[p] = {'cm': [0], 'recall': 0, 'prec': 0, 'false_pos': 0}

    for p in tqdm(pvals):
        details = np.array([[0,0], [0,0]])
        for i in range(len(attack_list)):
            attack_list[i] = alert_by_kde(attack_list[i], d, p)
            attack_list[i]['alert_window'] = attack_list[i].predicted_attack.rolling(3).sum() == 3
            details += confusion_matrix(attack_list[i]['actual_attack'], attack_list[i]['alert_window'])
        results_kde_final[p]['cm'] = details
        results_kde_final[p]['recall'] = details[1,1]/(details[1,1]+details[1,0])
        results_kde_final[p]['prec'] = details[1,1]/(details[1,1]+details[0,1])
        results_kde_final[p]['f1'] = 2*((results_kde_final[p]['prec']*results_kde_final[p]['recall'])/(results_kde_final[p]['prec']+results_kde_final[p]['recall']))
        results_kde_final[p]['false_pos'] = details[0,1]/(details[0,1]+details[0,0])
    
    picklify(results_kde_final, os.path.dirname(os.getcwd()) + "/results_kde_final_no_pretraining.pkl")


def get_results_gauss(attack_list, d):
    """
    Marks as attack when last three time_diffs had had p-value less than the p-value threshold for Gaussian distribution
    Calculates confusion matrix, precision, recall, false positive rate and saves to dictionary
    """

    pvals_gauss = sorted(list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)))
    # pvals_gauss = list(np.arange(0.01, 0.21, 0.01))
    # pvals_gauss = list(np.arange(0.0001, 0.0011, 0.0001))

    results_gauss_final = {}
    for p in pvals_gauss:
        results_gauss_final[p] = {'cm': [0], 'recall': 0, 'prec': 0, 'false_pos': 0}

    for p in tqdm(pvals_gauss):
        details = np.array([[0,0], [0,0]])
        for i in range(len(attack_list)):
            attack_list[i] = alert_by_gauss(attack_list[i], d, p)
            attack_list[i]['alert_window'] = attack_list[i].predicted_attack.rolling(3).sum() == 3
            details += confusion_matrix(attack_list[i]['actual_attack'], attack_list[i]['alert_window'])

        #print("p: ", p)
        print(details)
        results_gauss_final[p]['cm'] = details
        results_gauss_final[p]['recall'] = details[1,1]/(details[1,1]+details[1,0])
        results_gauss_final[p]['prec'] = details[1,1]/(details[1,1]+details[0,1])
        results_gauss_final[p]['f1'] = 2*((results_gauss_final[p]['prec']*results_gauss_final[p]['recall'])/(results_gauss_final[p]['prec']+results_gauss_final[p]['recall']))
        results_gauss_final[p]['false_pos'] = details[0,1]/(details[0,1]+details[0,0])

    picklify(results_gauss_final, os.path.dirname(os.getcwd()) + "/results_gauss_final.pkl")


def get_results_gauss_no_pretraining(attack_list, d):
    """
    Marks as attack when last three time_diffs had had p-value less than the p-value threshold for Gaussian distribution
    Calculates confusion matrix, precision, recall, false positive rate and saves to dictionary
    """

    pvals_gauss = sorted(list(np.arange(0.001, 0.01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)))
    # pvals_gauss = list(np.arange(0.01, 0.21, 0.01))
    # pvals_gauss = list(np.arange(0.0001, 0.0011, 0.0001))

    results_gauss_final = {}
    for p in pvals_gauss:
        results_gauss_final[p] = {'cm': [0], 'recall': 0, 'prec': 0, 'false_pos': 0}

    for p in tqdm(pvals_gauss):
        details = np.array([[0,0], [0,0]])
        for i in range(len(attack_list)):
            attack_list[i] = alert_by_gauss(attack_list[i], d, p)
            attack_list[i]['alert_window'] = attack_list[i].predicted_attack.rolling(3).sum() == 3
            details += confusion_matrix(attack_list[i]['actual_attack'], attack_list[i]['alert_window'])

        #print("p: ", p)
        print(details)
        results_gauss_final[p]['cm'] = details
        results_gauss_final[p]['recall'] = details[1,1]/(details[1,1]+details[1,0])
        results_gauss_final[p]['prec'] = details[1,1]/(details[1,1]+details[0,1])
        results_gauss_final[p]['f1'] = 2*((results_gauss_final[p]['prec']*results_gauss_final[p]['recall'])/(results_gauss_final[p]['prec']+results_gauss_final[p]['recall']))
        results_gauss_final[p]['false_pos'] = details[0,1]/(details[0,1]+details[0,0])

    picklify(results_gauss_final, os.path.dirname(os.getcwd()) + "/results_gauss_final_no_pretraining.pkl")


def get_results_binning(attack_list, D, n=6):
    """
    Marks as attack when 6 messages with same aid come in less than mu*4 seconds (where mu is average time_diff for the aid)
    Calculates confusion matrix, precision, recall, false positive rate and saves to dictionary
    """

    ## Initialize results dictionary
    results_binning = {}
    for i in range(len(attack_list)):
        results_binning[i+1] = {'cm': [0], 'recall': 0, 'prec':0, 'false_pos':0}
    results_binning['total'] = {'cm': [0], 'recall': 0, 'prec':0, 'false_pos':0}

    for i in tqdm(range(len(attack_list))):
        results_binning[i+1]['cm'] = alert_by_bin(attack_list[i], D, n)
        #print(results_binning[i+1]["cm"])
        results_binning[i+1]['prec'] = results_binning[i+1]['cm'][1,1]/(results_binning[i+1]['cm'][1,1]+results_binning[i+1]['cm'][0,1])
        results_binning[i+1]['recall'] = results_binning[i+1]['cm'][1,1]/(results_binning[i+1]['cm'][1,1]+results_binning[i+1]['cm'][1,0])
        results_binning[i+1]['false_pos'] = results_binning[i+1]['cm'][0,1]/(results_binning[i+1]['cm'][0,1]+results_binning[i+1]['cm'][0,0])
        results_binning['total']['cm'] += results_binning[i+1]['cm']

    #print(results_binning)
    results_binning['total']['prec'] =  results_binning['total']['cm'][1,1]/(results_binning['total']['cm'][1,1]+results_binning['total']['cm'][0,1])
    results_binning['total']['recall'] = results_binning['total']['cm'][1,1]/(results_binning['total']['cm'][1,1]+results_binning['total']['cm'][1,0])
    results_binning['total']['f1'] = 2*((results_binning['total']['prec']*results_binning['total']['recall'])/(results_binning['total']['prec']+results_binning['total']['recall']))
    results_binning['total']['false_pos'] = results_binning['total']['cm'][0,1]/(results_binning['total']['cm'][0,1]+results_binning['total']['cm'][0,0])

    #return results_binning

    picklify(results_binning, os.path.dirname(os.getcwd()) + "/results_binning_final.pkl")


def get_results_binning_various_p(attack_list, D, n=6):
    """
    Marks as attack when 6 messages with same aid come in less than mu*4 seconds (where mu is average time_diff for the aid)
    Calculates confusion matrix, precision, recall, false positive rate and saves to dictionary
    """

    pvals_binning = np.linspace(1, 10, 19) # 0, 4

    ## Initialize results dictionary
    results_binning_final = {}
    for p in pvals_binning:
        results_binning_final[p] = {'cm': [0], 'recall': 0, 'prec': 0, 'false_pos': 0}

    for p in tqdm(pvals_binning):
        details = np.array([[0,0], [0,0]])
        for i in range(len(attack_list)):
            details += alert_by_bin_various_p(attack_list[i], D, p, n)
        results_binning_final[p]["cm"] = details
        results_binning_final[p]['prec'] = details[1,1]/(details[1,1]+details[0,1])
        results_binning_final[p]['recall'] = details[1,1]/(details[1,1]+details[1,0])
        results_binning_final[p]['f1'] = 2*((results_binning_final[p]['prec']*results_binning_final[p]['recall'])/(results_binning_final[p]['prec']+results_binning_final[p]['recall']))
        results_binning_final[p]['false_pos'] = details[0,1]/(details[0,1] + details[0,0])
  

    # return results_binning_final

    picklify(results_binning_final, os.path.dirname(os.getcwd()) + "/results_binning_final.pkl")

def get_results_binning_various_p_no_pretraining(attack_list, D, n=6):
    """
    Marks as attack when 6 messages with same aid come in less than mu*4 seconds (where mu is average time_diff for the aid)
    Calculates confusion matrix, precision, recall, false positive rate and saves to dictionary
    """

    pvals_binning = np.linspace(1, 10, 19) # 0, 4

    ## Initialize results dictionary
    results_binning_final = {}
    for p in pvals_binning:
        results_binning_final[p] = {'cm': [0], 'recall': 0, 'prec': 0, 'false_pos': 0}

    for p in tqdm(pvals_binning):
        details = np.array([[0,0], [0,0]])
        for i in range(len(attack_list)):
            details += alert_by_bin_various_p(attack_list[i], D, p, n)
        results_binning_final[p]["cm"] = details
        results_binning_final[p]['prec'] = details[1,1]/(details[1,1]+details[0,1])
        results_binning_final[p]['recall'] = details[1,1]/(details[1,1]+details[1,0])
        results_binning_final[p]['f1'] = 2*((results_binning_final[p]['prec']*results_binning_final[p]['recall'])/(results_binning_final[p]['prec']+results_binning_final[p]['recall']))
        results_binning_final[p]['false_pos'] = details[0,1]/(details[0,1] + details[0,0])
  

    # return results_binning_final

    picklify(results_binning_final, os.path.dirname(os.getcwd()) + "/results_binning_final_no_pretraining.pkl")


def get_results_binning_J1939(attack_list, D, n=6):
    """
    Marks as attack when 6 messages with same aid come in less than mu*4 seconds (where mu is average time_diff for the aid)
    Calculates confusion matrix, precision, recall, false positive rate and saves to dictionary
    """

    ## Initialize results dictionary
    results_binning = {}
    for i in range(len(attack_list)):
        results_binning[i+1] = {'cm': [0], 'recall': 0, 'prec':0, 'false_pos':0}
    results_binning['total'] = {'cm': [0], 'recall': 0, 'prec':0, 'false_pos':0}

    for i in range(len(attack_list)):
        try:
            results_binning[i+1]['cm'] = alert_by_bin_J1939(attack_list[i], D, n)
            results_binning[i+1]['prec'] = results_binning[i+1]['cm'][1,1]/(results_binning[i+1]['cm'][1,1]+results_binning[i+1]['cm'][0,1])
            results_binning[i+1]['recall'] = results_binning[i+1]['cm'][1,1]/(results_binning[i+1]['cm'][1,1]+results_binning[i+1]['cm'][1,0])
            results_binning[i+1]['false_pos'] = results_binning[i+1]['cm'][0,1]/(results_binning[i+1]['cm'][0,1]+results_binning[i+1]['cm'][0,0])
            results_binning['total']['cm'] += results_binning[i+1]['cm']
        except:
            pass
    try:
        results_binning['total']['prec'] =  results_binning['total']['cm'][1,1]/(results_binning['total']['cm'][1,1]+results_binning['total']['cm'][0,1])
        results_binning['total']['recall'] = results_binning['total']['cm'][1,1]/(results_binning['total']['cm'][1,1]+results_binning['total']['cm'][1,0])
        results_binning['total']['false_pos'] = results_binning['total']['cm'][0,1]/(results_binning['total']['cm'][0,1]+results_binning['total']['cm'][0,0])
    except:
        pass
    return results_binning

def print_details(y_true, y_hat, filename=''):
    cm = confusion_matrix(y_true, y_hat)
    prec, rec, fscore, _ = precision_recall_fscore_support(
        y_true, y_hat, average='binary', pos_label=1)

    if filename != '':
        with open(filename + "_" + ".txt", 'a+') as file:  # Use file to refer to the file object

            file.write("Confusion Matrix of %s is \n%r\n" % (name, cm))
            file.write(
                f"Prec = {prec:.4f}, recall= {rec:.4f}, fscore = {fscore:.4f}\n\n")
    else:
        print("Confusion Matrix is \n%r\n" % (cm))
        print(
            f"Prec = {prec:.4f}, recall= {rec:.4f}, fscore = {fscore:.4f}\n\n")
    return {'cm': cm, 'prec': prec, 'recall': rec, 'fscore': fscore}
