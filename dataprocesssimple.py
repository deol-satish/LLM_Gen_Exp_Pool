import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import math
import random
import pickle


# Open the input file and output file in the same context
with open('llmrawdata.txt', 'r') as infile, open('lmprocesseddata.txt', 'w') as outfile:
    for line in infile:
        if 'l4s_ecn_marking-start' in line:
            # Remove the prefix 'l4s_ecn_marking-start,' and trailing ' end'
            parts = line.split("-")
            if len(parts) > 1:
                content = parts[1].strip()  # Remove any leading/trailing whitespace
                if content.endswith('end'):
                    # Write the content to the output file, stripping the 'end' part
                    outfile.write(content[6:-4] + '\n')
column_list = [
    "queue_type",                   # q->queue_type
    "qdelay_reference",             # pprms->qdelay_ref
    "tupdate",                      # pprms->tupdate
    "max_burst",                    # pprms->max_burst
    "max_ecn_threshold",            # pprms->max_ecnth
    "alpha_coefficient",            # pprms->alpha
    "beta_coefficient",             # pprms->beta
    "flags",                        # pprms->flags
    "burst_allowance",              # pst->burst_allowance
    "drop_probability",             # pst->drop_prob
    "current_queue_delay",          # pst->current_qdelay
    "previous_queue_delay",         # pst->qdelay_old
    "accumulated_probability",      # pst->accu_prob
    "measurement_start_time",       # pst->measurement_start
    "average_dequeue_time",         # pst->avg_dq_time
    "dequeue_count",                # pst->dq_count
    "status_flags",                 # pst->sflags
    "total_packets",                # q->stats.tot_pkts
    "total_bytes",                  # q->stats.tot_bytes
    "queue_length",                 # q->stats.length
    "length_in_bytes",              # q->stats.len_bytes
    "total_drops",                  # q->stats.drops
    "dequeue_action",               # dequeue_action
]


df=pd.read_csv("lmprocesseddata.txt",names=column_list,header=None)

# Drop columns that contain 'pprms' in their name
columns_to_drop = [
    "qdelay_reference",             # pprms->qdelay_ref
    "tupdate",                      # pprms->tupdate
    "max_burst",                    # pprms->max_burst
    "max_ecn_threshold",            # pprms->max_ecnth
    "alpha_coefficient",            # pprms->alpha
    "beta_coefficient",             # pprms->beta
    "flags",                        # pprms->flags
    ]
df = df.drop(columns=columns_to_drop)


df['dequeue_action']=df['dequeue_action']-1
df['dequeue_action'].unique()
df['dequeue_action'].value_counts()

class ExperiencePool:
    """
    Experience pool for collecting trajectories.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, reward, done):
        self.states.append(state)  # sometime state is also called obs (observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

# Define the list of columns to include
columns_to_use = [
    'queue_type', 
    'burst_allowance',
    'drop_probability',
    'current_queue_delay',
    'accumulated_probability',
    'average_dequeue_time',
    'total_bytes',
    'total_drops'
]


import pickle
exp_pool = ExperiencePool()
# Initialize the global reward variable
global_reward = 0

# Iterate through each row and update the global reward variable
for index, row in df.iterrows():
    state = np.array(row[columns_to_use], dtype=np.float32)
    exp_pool.add(state=state, action=row['dequeue_action'], reward=row['current_queue_delay'], done=0)
    if index > df.shape[0] *0.005:
        break;

pickle_save_path='exp_pool_l4s.pkl'
pickle.dump(exp_pool, open( pickle_save_path, 'wb'))
print(f"Done. Experience pool saved at:", pickle_save_path)