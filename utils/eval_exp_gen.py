# experience_generator.py
import numpy as np
import pickle
from utils.exp_pool import ExperiencePool
columns_to_use = [
    'queue_type', 'burst_allowance', 'drop_probability', 'current_queue_delay',
    'accumulated_probability', 'length_in_bytes', 'packet_length'
]

def gen_eval_exp_pool(df, pickle_save_path='exp_pool_l4s_eval.pkl', eval_exp_percent=0.02):
    """
    Generates the evaluation experience pool.
    """

    exp_pool = ExperiencePool()

    for index, row in df.iterrows():
        state = np.array(row[columns_to_use], dtype=np.float32)
        exp_pool.add(state=state, action=row['dequeue_action'], reward=row['current_queue_delay'], done=0)
        
        if index > df.shape[0] * eval_exp_percent:
            break

    with open(pickle_save_path, 'wb') as f:
        pickle.dump(exp_pool, f)
    
    print(f"Evaluation experience pool saved at: {pickle_save_path}")
    print("len(actions)", len(exp_pool.actions))
