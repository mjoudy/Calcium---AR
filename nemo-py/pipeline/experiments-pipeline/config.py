data_dir = '/work/ws/nemo/fr_mj200-lasso_reg-0/pipeline/source_data/t-60e6'
data_file_name = 'spikes-60e6-ms.npy'
spikes_address = data_dir+ '/' + data_file_name

chunks_nums = 100
chunks_dir = '/work/ws/nemo/fr_mj200-lasso_reg-0/pipeline/source_data/t-60e6/calcium-change/chunked'
data_chunk_name = data_file_name.split('.')[0]


do_plots = False
plot_neuron = 500

calcium_tau = 100
calcium_dir = '/work/ws/nemo/fr_mj200-lasso_reg-0/pipeline/source_data/t-60e6/calcium-change/chunked-calcium'


sg_win_len = 51
sg_pl_order = 3
sg_delta = 1.

cut_win_len = 5
processed_dir = "/work/ws/nemo/fr_mj200-lasso_reg-0/pipeline/source_data/t-60e6/calcium-change/chunked-processed"

feed_chunk_name = data_chunk_name.replace('spikes', f'feed_tau{calcium_tau}')

final_data_name = f"feed-60e6-tau{calcium_tau}-final_data.npy"


conn_matrix = 'connectivity-60e6-ms.npy'
inf_lag = 10
est_conn_name = final_data_name.replace("feed", f"est_conn_lag{inf_lag}")
est_conn_add = data_dir + "/" + est_conn_name


params_dict = {'Data Length': data_file_name.split('-')[1], 'Calcium tau': calcium_tau, 
               'sav-gol-win-len': sg_win_len, 'sav-gol-poly-order': sg_pl_order, 
               'sav-gol-delta': sg_delta, 'Length of cutting spikes win': cut_win_len,
               'Inference-lag': inf_lag}
