# experience_generator.py
import numpy as np
import pickle
from utils.exp_pool import ExperiencePool
columns_to_use = [
    'queue_type', 'burst_allowance', 'drop_probability', 'current_queue_delay',
    'accumulated_probability', 'length_in_bytes', 'packet_length'
]

def gen_train_exp_pool(df, pickle_save_path='exp_pool_l4s_train.pkl', train_exp_percent=0.2, window_size=5):
    """
    Generates the training experience pool.
    """
    exp_pool = ExperiencePool()
    prev_action = None

    for index, row in df.iterrows():
        state = np.array(row[columns_to_use], dtype=np.float32)
        current_action = row['dequeue_action']
        
        if prev_action is not None and current_action != prev_action:
            start_index = max(0, index - window_size)
            end_index = index
            selected_rows = df.iloc[start_index:end_index]
            for _, selected_row in selected_rows.iterrows():
                state = np.array(selected_row[columns_to_use], dtype=np.float32)
                exp_pool.add(state=state, action=selected_row['dequeue_action'], reward=selected_row['current_queue_delay'], done=0)
        
        if index > df.shape[0] * train_exp_percent:
            break
        
        prev_action = current_action

    with open(pickle_save_path, 'wb') as f:
        pickle.dump(exp_pool, f)
    
    print(f"Training experience pool saved at: {pickle_save_path}")
    print("len(actions)", len(exp_pool.actions))