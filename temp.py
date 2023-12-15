def hamming_distance_int_list(seq1, seq2):
    # Calculate the Hamming distance between two binary sequences represented as lists
    return sum(bin(x ^ y).count('1') for x, y in zip(seq1, seq2))

def string_hamming_distance(str1, str2):
    # Ensure both strings are of the same length. If not, pad the shorter one with zeros.
    string1 = ''.join(f'{item:03}' for item in str1)
    string2 = ''.join(f'{item:03}' for item in str2)
    print(string1, string2)

    # Calculate the Hamming distance
    distance = 0
    differing_indices = []
    for i, (ch1, ch2) in enumerate(zip(string1, string2)):
        if ch1 != ch2:
            distance += 1
            differing_indices.append(i)
    
    # Extract start and end index of differing region
    if differing_indices:
        start_index = differing_indices[0]
        end_index = differing_indices[-1]
    else:
        start_index, end_index = 0, 0
    
    return distance, int(start_index/3), int(end_index/3)

def all_pair_hamming_distance_int_list(sequence_list):
    n = len(sequence_list)
    b_max_distance = 0
    b_min_distance = 64
    s_max_distance = 0
    s_min_distance = 24
    r_start = 7
    r_end = 0
    for i in range(n):
        for j in range(i + 1, n):
            b_distance = hamming_distance_int_list(sequence_list[i], sequence_list[j])
            s_distance, start, end = string_hamming_distance(sequence_list[i], sequence_list[j])
            b_max_distance = max(b_max_distance, b_distance)
            b_min_distance = min(b_min_distance, b_distance)
            s_max_distance = max(s_max_distance, s_distance)
            s_min_distance = min(s_min_distance, s_distance)
            r_start = min(r_start, start)
            r_end = max(r_end, end)
            # Chỉ so sánh với 00000000000000 rồi trừ ra
            # print(f"{max_distance} and {min_distance}")
            # aa
            
    return b_max_distance, b_min_distance, s_max_distance, s_min_distance, r_start, r_end

def get_unique_list(can_dict):
    for can_id in can_dict:
        payload_unique = []
        for payload in can_dict[can_id]['payload']:
            if payload not in payload_unique:
                payload_unique.append(payload)    
        can_dict[can_id]['uni_payload'] = payload_unique
        del can_dict[can_id]['payload']
        # print(f"CANID: {can_id} ori: {len(can_dict[can_id]['payload'])} with {len(can_dict[can_id]['uni_payload'])}")
        
    return can_dict

def calc_message_dhd_threshold(can_dict):
    can_dict = get_unique_list(can_dict)
    json.dump(can_dict, open('/home/tiendat/transformer-entropy-ids/road/candict.txt', 'w'))
    for can_id in can_dict:
        # print(f"\n{can_id}:{len(can_dict[can_id]['uni_payload'])} processing...")
        # start_time = time.time()
        if len(can_dict[can_id]['uni_payload']) > 1:
            b_max_distance, b_min_distance, s_max_distance, s_min_distance, start, end = all_pair_hamming_distance_int_list(can_dict[can_id]['uni_payload'])
            can_dict[can_id]['b_max_bhd'] = b_max_distance
            can_dict[can_id]['b_min_bhd'] = b_min_distance
            can_dict[can_id]['s_max_bhd'] = s_max_distance
            can_dict[can_id]['s_min_bhd'] = s_min_distance
            can_dict[can_id]['start'] = start
            can_dict[can_id]['end'] = end
        else:
            can_dict[can_id]['b_max_bhd'] = 0
            can_dict[can_id]['b_min_bhd'] = 0
            can_dict[can_id]['s_max_bhd'] = 0
            can_dict[can_id]['s_min_bhd'] = 0
            can_dict[can_id]['start'] = 0
            can_dict[can_id]['end'] = 7

        # del can_dict[can_id]['uni_payload']
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"CANID: {can_id} max distance: {can_dict[can_id]['max_bhd']} and min distance: {can_dict[can_id]['min_bhd']} on {execution_time} seconds ==")
    return can_dict

def calc_timediff_threshold(can_dict):
    for canid_i in can_dict:
        # print(f"Process {canid_i}")
        max_time_diff = max(can_dict[canid_i]['timediff'])
        min_time_diff = min(can_dict[canid_i]['timediff'])
        can_dict[canid_i]['max_time'] = max_time_diff
        can_dict[canid_i]['min_time'] = min_time_diff
        del can_dict[canid_i]['timediff']
        # print(f"CANID: {canid_i} with max: {can_dict[canid_i]['max_time']} and min: {can_dict[canid_i]['min_time']}")
    return can_dict

def calc_anomaly_threshhold(df_anomaly):
    can_dict = {}
    canid = []
    message = []
    time_diff = [] 
    for msg in df_anomaly['canID']:
        canid.append(msg)
    for msg in df_anomaly['Data']:
        message.append(msg)
    for msg in df_anomaly['TimeDiffs']:
        time_diff.append(msg)
    
    for i, id in enumerate(canid):
        if id in can_dict:
            can_dict[id]['payload'].append(message[i])
            can_dict[id]['timediff'].append(time_diff[i])
        else:
            can_dict[id] = {'payload': [message[i]], 'timediff': [time_diff[i]]}
    
    print("Calculating threshold...................")
    can_dict = calc_timediff_threshold(can_dict)
    can_dict = calc_message_dhd_threshold(can_dict)
    print(f"DONE: Included {len(can_dict.keys())} canID")
    for can_id in can_dict:
        print(f"\nCANID {can_id}:")
        print(f"Max time interval: {can_dict[can_id]['max_time']} and min time interval: {can_dict[can_id]['min_time']}")
        print(f"Max hamming {can_dict[can_id]['max_bhd']} and min hamming {can_dict[can_id]['min_bhd']}")
        
    return can_dict



import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

df_dis = df_all_normal[df_all_normal['aid'] == 6]

# # Set up the figure
# plt.figure(figsize=(8, 6))

# # plt.hist(df_dis['time_diffs'], bins=50, edgecolor='black', alpha=0.8, label='Histogram')
# plt.scatter(df['time_diffs'], c='blue', marker='o', label='Data Points')

# plt.xlim(left=0, right=200)
# plt.ylim(bottom=0, top=10)
# plt.title('Distribution of time diffs')
# plt.xlabel('Time diffs')
# plt.ylabel('Density')
# plt.legend()
# plt.show()

# # Count the frequency of each unique value in the 'age' column
# time_counts = df_dis['time_diffs'].value_counts()
# display(time_counts)

# # Take the top 5 most frequent ages for demonstration
# top_5_ages = time_counts.nlargest(5)

# # Add a 'Others' category for all other ages
# top_5_ages['Others'] = time_counts.sum() - top_5_ages.sum()

# # Create a pie chart for the 'age' column
plt.figure(figsize=(10, 7))
# plt.pie(top_5_ages, labels=top_5_ages.index, autopct='%1.1f%%', startangle=90)
# plt.title('Pie Chart of Top 5 Most Frequent Ages and Others')
# plt.show()

# Count the frequency of each unique age
age_counts = df_dis['time_diffs'].value_counts().reset_index()
age_counts.columns = ['time_diffs', 'Frequency']

# Create a bubble chart for the 'age' column using frequencies as the size of the bubbles
plt.scatter(age_counts['time_diffs'], np.zeros_like(age_counts['time_diffs']), c='blue', 
            s=age_counts['Frequency']*10, alpha=0.6, label='Data Points')

# Add labels, title, and legend
plt.xlabel('Time')
plt.title('Bubble Chart of Age (Bubble Size Proportional to Frequency)')
plt.yticks([])  # Hide y-axis
plt.legend()

# Show the plot
plt.show()


