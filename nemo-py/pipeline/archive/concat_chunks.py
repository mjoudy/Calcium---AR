import os
import numpy as np


# Function to load chunks and concatenate them
def load_and_concatenate_chunks(prefix, num_chunks, load_dir, axis=1):
    concatenated_data = []
    for i in range(num_chunks):
        filename = os.path.join(load_dir, f'{prefix}_{i}.npy')
        chunk = np.load(filename)
        concatenated_data.append(chunk)
    print('the first chunk shape is:' + str(concatenated_data[0].shape))
    return np.concatenate(concatenated_data, axis=axis)

# Specify the directory where the processed chunks are saved
#processed_data_dir = os.getcwd()
#processed_data_dir = os.path.dirname(__file__)
processed_data_dir = "/work/ws/nemo/fr_mj200-lasso_reg-0/pipeline/source_data/t-60e6/chunked-processed"

data_name = 'spikes-10e4-ms'
chunks_nums = 10
# Load and concatenate the processed chunks along axis=1
concatenated_data = load_and_concatenate_chunks(data_name, chunks_nums, processed_data_dir, axis=1)
print('the whole signals shape is:' + str(concatenated_data.shape))
# Specify the directory and filename for the final concatenated file
final_data_dir = os.path.dirname(__file__)
final_data_file = 'final_data.npy'
final_data_address = os.path.join(final_data_dir, final_data_file)

# Save the final concatenated data
np.save(final_data_address, concatenated_data)
