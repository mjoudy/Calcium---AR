import numpy as np

data_address = 'spikes-10e4-ms.npy'
data = np.load(data_address)

def divide_to_chunks(array, num_chunks):
    chunk_size = array.shape[0] // num_chunks
    chunks = [array[i:i+chunk_size] for i in range(0, array.shape[0], chunk_size)]
    return chunks

def save_chunks(chunks, prefix):
    for i, chunk in enumerate(chunks):
        np.save(f'{prefix}_{i}.npy', chunk)

data_name = data_address.split('.')[0]
chunks = divide_to_chunks(data, 10)
save_chunks(chunks, data_name)

