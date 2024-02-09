import chunk_funcs as cf
import concat_chunks as cc
import os
import subprocess as sbp
import conn_inf as ci
import utility_pipeline as up

#this part of address is for NEMO:''/work/ws/nemo/fr_mj200-lasso_reg-0/pipeline/'
#the rest is for my package
input_dir = '/work/ws/nemo/fr_mj200-lasso_reg-0/pipeline/source_data/t-60e6'
input_file_name = 'spikes-60e6-ms.npy'
#chunked_dir = input_dir/chunked
chunks_nums = 100
cf.chunk_data(input_dir, input_file_name, number_of_chunks=chunks_nums)

#need to be modified for a general address
current_dir = os.path.dirname(os.path.abspath(__file__))
calcium_sim_sh = os.path.join(current_dir, 'calcium_sim.sh')
#submit_command_calcium = ['msub', calcium_sim_sh]
#sbp.run(submit_command_calcium, check=True, capture_output=True, text=True)
calcium_job_id = up.submit_job_and_get_id(calcium_sim_sh)
if calcium_job_id:
    print(f'calcium simulation job submitted with job id {calcium_job_id}')
    up.check_job_status(calcium_job_id)
else:
    print('calcium simulation job submission failed')

preprocess_sh = os.path.join(current_dir, 'preprocess.sh')
#submit_command_preprocess = ['msub', preprocess_sh]
#sbp.run(submit_command_preprocess, check=True, capture_output=True, text=True)
preprocess_job_id = up.submit_job_and_get_id(preprocess_sh)
if preprocess_job_id:
    print(f'preprocess job submitted with job id {preprocess_job_id}')
    up.check_job_status(preprocess_job_id)
else:
    print('preprocess job submission failed')

concat_path = os.path.join(current_dir, 'chunked-processed')
chunk_prefix = 'feed-60e6-ms'
final_data_file = chunk_prefix + '-final_data.npy'
final_data_address = os.path.join(input_dir, final_data_file)
cc.load_and_concatenate_chunks(chunk_prefix, chunks_nums, concat_path, final_data_address)

conn_matrix = 'connectivity-60e6-ms.npy'
feed_data = 'feed-60e6-ms-final_data.npy'
ci.conn_inf(input_dir, conn_matrix, final_data_address, feed_data)