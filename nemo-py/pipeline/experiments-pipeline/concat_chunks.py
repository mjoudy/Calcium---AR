import os
import numpy as np
from loging_pipeline import log_process
from config import *


def load_and_concatenate_chunks(prefix, num_chunks, load_dir, axis=1):
    concatenated_data = []
    for i in range(num_chunks):
        filename = os.path.join(load_dir, f'{prefix}_{i}.npy')
        chunk = np.load(filename)
        concatenated_data.append(chunk)
    print('the first chunk shape is:' + str(concatenated_data[0].shape))
    return np.concatenate(concatenated_data, axis=axis)



# Load and concatenate the processed chunks along axis=0
concatenated_data = load_and_concatenate_chunks(feed_chunk_name, chunks_nums, processed_dir, axis=1)
print('the whole signals shape is:' + str(concatenated_data.shape))


final_data_address = os.path.join(data_dir, final_data_name)

np.save(final_data_address, concatenated_data)

log_process("preprocessed signals of chunks concatenated.")
