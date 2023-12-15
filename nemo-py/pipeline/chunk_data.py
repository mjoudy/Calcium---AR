import os
import numpy as np

#data_dir = '/home/fr/fr_fr/fr_mj200'
data_dir = '/work/ws/nemo/fr_mj200-lasso_reg-0/pipeline/source_data/t-60e6'
#data_dir = os.getpwd()
#data_dir = os.path.dirname(__file__)
print(data_dir)
data_file = 'spikes-60e6-ms.npy'
data_address = data_dir+ '/' + data_file
data = np.load(data_address)

print('the whole signals shape is:' + str(data.shape))

def divide_to_chunks(array, num_chunks, axis=1):
    chunk_size = array.shape[axis] // num_chunks
    chunks = [array.take(range(i*chunk_size, (i+1)*chunk_size), axis=axis) for i in range(num_chunks)]
    return chunks

'''
def divide_to_chunks(array, num_chunks):
    chunk_size = array.shape[0] // num_chunks
    chunks = [array[i:i+chunk_size] for i in range(0, array.shape[0], chunk_size)]
    return chunks
'''

def save_chunks(chunks, prefix, save_dir):
    for i, chunk in enumerate(chunks):
        filename = os.path.join(save_dir, f'{prefix}_{i}.npy')
        np.save(filename, chunk)


chunks_nums = 100
chunks = divide_to_chunks(data, chunks_nums)

data_name = data_file.split('.')[0]
save_dir = '/work/ws/nemo/fr_mj200-lasso_reg-0/pipeline/source_data/t-60e6/chunked'
save_chunks(chunks, data_name, save_dir)

print('the first chunk shape is:' + str(chunks[0].shape))
print(save_dir)
