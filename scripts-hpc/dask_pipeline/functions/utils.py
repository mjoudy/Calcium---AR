import json

#This function has been created to avoid confiosion of files durind mass experiments.
#One can use it to name files in a way that is easy to understand and identify parameters space and stages of the job(spikes, calcium, preprocessed, ...).
#this function can be easily updated by adding more arguments as parameters of different stages. 
def nameit(replacement, **kwargs):
    with open('temp_names.json', 'r') as file:
        file_names = json.load(file)
        sp_name = file_names['sp_file']

    base_name, ext = sp_name.rsplit('.zarr', 1)
    modified_name = base_name.replace('spikes', replacement)

    for key, value in kwargs.items():
        modified_name += f'-{key}{value}'

    modified_name += '.zarr'
    return modified_name
        
