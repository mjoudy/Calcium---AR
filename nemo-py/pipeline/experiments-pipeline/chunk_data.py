import os
import numpy as np
from loging_pipeline import *
from config import *

print(data_dir)

data = np.load(spikes_address)

print('the whole signals shape is:' + str(data.shape))

if os.path.exists('chunk_name.txt'):
    os.remove('chunk_name.txt')

if os.path.exists('cal_chunk_name.txt'):
    os.remove('cal_chunk_name.txt')


def divide_to_chunks(array, num_chunks, axis=1):
    chunk_size = array.shape[axis] // num_chunks
    chunks = [array.take(range(i*chunk_size, (i+1)*chunk_size), axis=axis) for i in range(num_chunks)]
    return chunks

def save_chunks(chunks, prefix, save_dir):
    for i, chunk in enumerate(chunks):
        filename = os.path.join(save_dir, f'{prefix}_{i}.npy')
        np.save(filename, chunk)



chunks = divide_to_chunks(data, chunks_nums)

save_chunks(chunks, data_chunk_name, chunks_dir)

print('the first chunk shape is:' + str(chunks[0].shape))
print(chunks_dir)


with open ('chunk_name.txt', 'w') as f:
    f.write(data_chunk_name)

with open ('cal_chunk_name.txt') as ff:
    ff.write(data_chunk_name.replace('spikes', f'calcium_tau{calcium_tau}'))

write_dict_to_json(params_dict, 'experiments_log.json')

log_process("chunking data is done")