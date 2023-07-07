import pandas as pd
# import vaex
import numpy as np
import glob
import dask.dataframe as dd
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import sys

class ACNode:
    def __init__(self):
        self.children = {}  # Dictionary to store child nodes
        self.is_end_of_pattern = False  # Flag to indicate end of a pattern
        self.failure = None  # Failure transition
        self.output = []  # Output labels

class AhoCorasick:
    def __init__(self):
        self.root = ACNode()  # Root of the trie

    def add_pattern(self, pattern):
        node = self.root
        for char in pattern:
            if char not in node.children:
                node.children[char] = ACNode()
            node = node.children[char]
        node.is_end_of_pattern = True
        node.output.append(pattern)

    def build_failure_transitions(self):
        queue = []

        # Initialize failure transitions of depth-1 nodes
        for child in self.root.children.values():
            child.failure = self.root
            queue.append(child)

        # Build failure transitions using BFS
        while queue:
            current = queue.pop(0)

            for char, child in current.children.items():
                queue.append(child)
                failure_state = current.failure

                while failure_state != self.root and char not in failure_state.children:
                    failure_state = failure_state.failure

                if char in failure_state.children:
                    child.failure = failure_state.children[char]
                else:
                    child.failure = self.root

                # Propagate output labels
                child.output += child.failure.output

    def match_patterns(self, text):
        current = self.root
        matches = []

        for char in text:
            while current != self.root and char not in current.children:
                current = current.failure

            if char in current.children:
                current = current.children[char]
            else:
                current = self.root

            # Output matching patterns
            if current.output:
                matches.extend(current.output)

        return matches

def fill_flag(sample):
    if not isinstance(sample['Flag'], str):
        col = 'Data' + str(sample['DLC'])
        sample['Flag'] = sample[col]
    return sample


def convert_canid_bits(cid):
    try:
        s = bin(int(str(cid), 16))[2:].zfill(29)
        bits = list(map(int, list(s)))
        return bits

    except:
        return None


attributes = ['Timestamp', 'canID', 'DLC',
                           'Data0', 'Data1', 'Data2',
                           'Data3', 'Data4', 'Data5',
                           'Data6', 'Data7', 'Flag']
dataset_path = './data/car-hacking/'
attack_types = ['DoS', 'Fuzzy', 'gear', 'RPM']
# attack = attack_types[1]
# file_name = '{}{}_dataset.csv'.format(dataset_path, attack)
# print(file_name)

canIDs = []
rule_based_set = {}

def plot_chart(pd_df):
    false_data = pd_df[pd_df['Flag'] == False]
    true_data = pd_df[pd_df['Flag'] == True]

    # Plotting the chart
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    ax.scatter(false_data['Timestamp'], false_data['canID'], color='red', label='False',
               s=5)  # Adjust the 's' parameter to change point size
    ax.scatter(true_data['Timestamp'], true_data['canID'], color='blue', label='True',
               s=5)  # Adjust the 's' parameter to change point size

    ax.set_xlabel('Timestamp')
    ax.set_ylabel('canID')
    ax.set_title('canID by Timestamp')
    ax.legend(loc="best")
    plt.show()


def extract_canID(file_name, canIDs, ac, rules, attack):
    df = dd.read_csv(file_name, header=None, names=attributes, dtype={'Data2': 'object', 'Data6': 'object', 'Data4': 'object', 'Data1': 'object'})
    print('Reading from {}: DONE'.format(file_name))
    print('Dask processing: -------------')
    df = df.apply(fill_flag, axis=1, meta={'Timestamp': 'float64', 'canID': 'object', 'DLC': 'int64', 'Data0': 'object', 'Data1': 'object', 'Data2': 'int64', 'Data3': 'object', 'Data4': 'object', 'Data5': 'object', 'Data6': 'float64', 'Data7': 'object', 'Flag': 'object'})
    pd_df = df.compute()
    pd_df = pd_df[['Timestamp', 'canID', 'Flag', 'DLC']].sort_values('Timestamp',  ascending=True)
    # pd_df['canBits'] = pd_df.canID.apply(convert_canid_bits)
    pd_df['Flag'] = pd_df['Flag'].apply(lambda x: True if x == 'T' else False)
    # plot_chart(pd_df)
    filtered_df = pd_df[pd_df['Flag'] == False]
    for canID, value in rules.items():
        time_interval = []
        last_timestamp = 0
        for index, row in filtered_df.iterrows():
            if row['canID'] == canID:
                if last_timestamp == 0:
                    last_timestamp = row['Timestamp']  # get first element
                    continue
                timestamp = row['Timestamp'] - last_timestamp  # calc delta t
                last_timestamp = row['Timestamp']  # update time stamp
                time_interval.append(timestamp)  # add delta t into array

        # print("Time interval of: ", canID, " is ", time_interval)
        time_file_name = "./data/time-interval-data/" + attack + "_" + str(canID) + ".txt"
        with open(time_file_name, 'w') as file:
            # Iterate over the array elements and write them to the file
            for item in time_interval:
                file.write(str(item) + '\n')
        # canIDs.append(row['canID'])
        # matches = ac.match_patterns(row['canID'])
        # if matches:
        #     if rule_based_set.get(row['canID']) is None:
        #         print(row['canID'], " added with ", str(row['DLC']))
        #     elif rule_based_set[row['canID']] != row['DLC']:
        #         print(row['canID'], " mapped with ", str(row['DLC']))
        #     rule_based_set[row['canID']] = row['DLC']

    # print(list(set(canIDs)))




def extract_DLC(file_name):
    df = dd.read_csv(file_name, header=None, names=attributes, dtype={'Data2': 'object', 'Data6': 'object', 'Data4': 'object', 'Data1': 'object'})
    print('Reading from {}: DONE'.format(file_name))
    print('Dask processing: -------------')
    df = df.apply(fill_flag, axis=1, meta={'Timestamp': 'float64', 'canID': 'object', 'DLC': 'int64', 'Data0': 'object', 'Data1': 'object', 'Data2': 'int64', 'Data3': 'object', 'Data4': 'object', 'Data5': 'object', 'Data6': 'float64', 'Data7': 'object', 'Flag': 'object'})
    pd_df = df.compute()
    pd_df = pd_df[['Timestamp', 'canID', 'Flag']].sort_values('Timestamp',  ascending=True)
    # pd_df['canBits'] = pd_df.canID.apply(convert_canid_bits)
    pd_df['Flag'] = pd_df['Flag'].apply(lambda x: True if x == 'T' else False)
    filtered_df = pd_df[pd_df['Flag'] == False]

    # for index, row in filtered_df.iterrows():
    #     print(row['canBits'])

    # print(list(set(canIDs)))


ac = AhoCorasick()

# Example usage
rules = {'0002': 8, '00a0': 8, '00a1': 8, '0105': 6, '0130': 8, '0131': 8, '0140': 8, '0153': 8, '018f': 8, '01f1': 8, '0260': 8, '02a0': 8, '02b0': 5, '02c0': 8, '0316': 8, '0329': 8, '0350': 8, '0370': 8, '0430': 8, '043f': 8, '0440': 8, '04b1': 8, '04f0': 8, '0545': 8, '05a0': 8, '05a2': 8, '05f0': 2, '0608': 8, '0690': 8, '06a1': 8, '06aa': 5, '071b': 8, '07cf': 8, '07d9': 8, '07da': 8, '07de': 8, '07e8': 8, '07e9': 8}

# Add rules to the Aho-Corasick trie
for rule, dlc in rules.items():
    ac.add_pattern(rule)

# Build failure transitions
ac.build_failure_transitions()

for attack in attack_types:
    file_name = '{}{}_dataset.csv'.format(dataset_path, attack)
    print(file_name)
    extract_canID(file_name, canIDs, ac, rules, attack)

start_time = time.time()

# Match the input data against the rules
normal = 0
attack = 0
# for canID in canIDs:
#     matches = ac.match_patterns(canID)
#     if matches:
#         normal += 1
#     else:
#         attack += 1

print("Rule based set: ", rule_based_set)

print("Normal canID:", normal, "seconds")
print("Attack canID:", attack, "seconds")

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# Measure memory usage
memory_usage = sys.getsizeof(ac)
print("Memory usage:", memory_usage, "bytes")