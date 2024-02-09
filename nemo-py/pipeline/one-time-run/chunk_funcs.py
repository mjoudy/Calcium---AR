import os
import numpy as np
from loging_pipeline import log_process

def load_data(data_dir, data_file):

    data_address = os.path.join(data_dir, data_file)
    print('input data address: ', data_address)
    try:
        data = np.load(data_address)
        print('the whole signals shape is:' + str(data.shape))
    except Exception as e:
        print('Failed to load data')
        raise
    return data

def divide_to_chunks(array, num_chunks, axis=1):
    chunk_size = array.shape[axis] // num_chunks
    chunks = [array.take(range(i*chunk_size, (i+1)*chunk_size), axis=axis) for i in range(num_chunks)]
    return chunks

def save_chunks(chunks, prefix, save_dir):
    for i, chunk in enumerate(chunks):
        filename = os.path.join(save_dir, f'{prefix}_{i}.npy')
        np.save(filename, chunk)

def chunk_data(data_dir, data_file, chunks_nums):
    data = load_data(data_dir, data_file)
    chunks = divide_to_chunks(data, chunks_nums)
    data_name = data_file.split('.')[0]
    save_dir = os.path.join(data_dir, 'chunked')
    save_chunks(chunks, data_name, save_dir)
    print('the first chunk shape is:' + str(chunks[0].shape))
    print(save_dir)
    log_process("chunking data is done")

