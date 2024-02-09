import os
import numpy as np
from loging_pipeline import log_process

# Function to load chunks and concatenate them
def load_and_concatenate_chunks(prefix, num_chunks, load_dir, final_add, axis=1):
    concatenated_data = []
    for i in range(num_chunks):
        filename = os.path.join(load_dir, f'{prefix}_{i}.npy')
        chunk = np.load(filename)
        concatenated_data.append(chunk)
    
    print('the first chunk shape is:' + str(concatenated_data[0].shape))
    final_data = np.concatenate(concatenated_data, axis=axis)
    print('the whole signals shape is:' + str(final_data.shape))
    np.save(final_add, final_data)
    log_process("preprocessed signals of chunks concatenated.")

