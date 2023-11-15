import os
import numpy as np

data_dir = '/home/fr/fr_fr/fr_mj200/'
data_file = 'spikes-10e4-ms.npy'
#data_address = data_dir+data_file
data_address = 'spikes-10e4-ms.npy'
data = np.load(data_address)

def divide_to_chunks(array, num_chunks):
    chunk_size = array.shape[0] // num_chunks
    chunks = [array[i:i+chunk_size] for i in range(0, array.shape[0], chunk_size)]
    return chunks

def save_chunks(chunks, prefix, save_dir):
    for i, chunk in enumerate(chunks):
        filename = os.path.join(save_dir, f'{prefix}_{i}.npy')
        np.save(filename, chunk)


chunks_nums = 10
chunks = divide_to_chunks(data, chunks_nums)

data_name = data_file.split('.')[0]
#save_dir = '/work/ws/nemo/fr_mj200-lasso_reg-0/pipeline/'
save_dir = os.getcwd()
save_chunks(chunks, data_name, save_dir)


