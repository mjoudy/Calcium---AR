import nest
import nest.raster_plot
import matplotlib.pyplot as plt
import numpy as np
import time
import h5py
import json
import os
#import yaml

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

base_path = os.environ.get("PROJECT_BASE_DIR", config['base_dir'])
config_path = os.path.join(base_path, config['network_config.json'])
functions_dir = os.path.join(base_path, config['functions_dir'])

class BrunelNetwork:

    def __init__(self, config_file='network_config.json'):

        with open(config_file, 'r') as f:
            params = json.load(f)

        self.sim_params = params['sim_params']
        self.network_params = params['network_params']
        self.syn_params = params['syn_params']
        self.neuron_params = params['neuron_params']

        self.NE = 4 * self.network_params['order']
        self.NI = 1 * self.network_params['order']
        self.N_neurons = self.NE + self.NI

        self.CE = int(self.network_params['epsilon'] * self.NE)
        self.CI = int(self.network_params['epsilon'] * self.NI)
        self.C_tot = int(self.CI + self.CE)

        self.J_ex = self.syn_params['J']
        self.J_in = -self.network_params['g'] * self.J_ex

        self.spikes_trains = None
        self.adj_matrix = None

    
    def ext_input(self):

        nu_th = self.neuron_params['V_th'] / (self.J_ex * self.CE * self.neuron_params['tau_m'])
        nu_ex = self.network_params['eta'] * nu_th
        p_rate = 10000.0 * nu_ex * self.CE

        return p_rate
    

    def setup_kernel(self):

        nest.ResetKernel()
        nest.SetKernelStatus({
            "local_num_threads": self.sim_params['cpu_num'],
            "resolution": self.sim_params['time_res'],
            "print_time": True,
            "overwrite_files": True
        })


    def setup_elements(self):
        
        nest.SetDefaults("iaf_psc_delta", self.neuron_params)
        nest.SetDefaults("poisson_generator", {"rate": self.ext_input()})
        nest.SetDefaults("dc_generator", {"amplitude": 6.})

        self.nodes_ex = nest.Create("iaf_psc_delta", self.NE, params=self.neuron_params)
        self.nodes_in = nest.Create("iaf_psc_delta", self.NI, params=self.neuron_params)

        self.noise = nest.Create("poisson_generator")

        self.multimeter_network_E = nest.Create("multimeter")
        nest.SetStatus(self.multimeter_network_E, {"record_from":["V_m"]})

        self.multimeter_network_I = nest.Create("multimeter")
        nest.SetStatus(self.multimeter_network_I, {"record_from":["V_m"]})

        nest.CopyModel("static_synapse", "excitatory",
                       {"weight": self.J_ex, "delay": self.syn_params['delay']})
        nest.CopyModel("static_synapse", "inhibitory",
                       {"weight": self.J_in, "delay": self.syn_params['delay']})
        
        self.sp_detector = nest.Create("spike_recorder")
        self.raster_exc = nest.Create("spike_recorder")
        self.raster_inh = nest.Create("spike_recorder")

    def connect_elements(self):

        nest.Connect(self.noise, self.nodes_ex, syn_spec="excitatory")
        nest.Connect(self.noise, self.nodes_in, syn_spec="excitatory")

        nest.Connect(self.multimeter_network_E, self.nodes_ex)
        nest.Connect(self.multimeter_network_I, self.nodes_in)

        nest.Connect(self.nodes_ex + self.nodes_in, self.sp_detector, syn_spec="excitatory")        

        nest.Connect(self.nodes_ex[:self.sim_params['N_rec']], self.raster_exc, syn_spec="excitatory")
        nest.Connect(self.nodes_in[:self.sim_params['N_rec']], self.raster_inh, syn_spec="excitatory")

        conn_params_ex = {'rule': 'fixed_indegree', 'indegree': self.CE}
        nest.Connect(self.nodes_ex, self.nodes_ex + self.nodes_in, conn_params_ex, "excitatory")

        conn_params_in = {'rule': 'fixed_indegree', 'indegree': self.CI}
        nest.Connect(self.nodes_in, self.nodes_ex + self.nodes_in, conn_params_in, "inhibitory")


    def simulate(self):

        start_sim = time.time()
        print("Starting Brunel network simulation: ", start_sim)
        nest.Simulate(self.sim_params['sim_length'])
        end_sim = time.time()
        print("Simulation finished: ", end_sim)

        self.events_ex = nest.GetStatus(self.raster_exc, keys="n_events")[0]
        self.events_in = nest.GetStatus(self.raster_inh, keys="n_events")[0]

        rates_ex = self.events_ex / self.sim_params['sim_length'] * 1000.0 / self.sim_params['N_rec']
        rates_in = self.events_in / self.sim_params['sim_length'] * 1000.0 / self.sim_params['N_rec']

        num_synapses = (nest.GetDefaults("excitatory")["num_connections"] +
                nest.GetDefaults("inhibitory")["num_connections"])

        sim_time = end_sim - start_sim

        print("Brunel network simulation finished")
        print("Number of neurons: {}".format(self.N_neurons))
        print("Number of synapses: {}".format(num_synapses))
        
        print("Excitatory rate: %.2f Hz".format(rates_ex))
        print("Inhibitory rate: %.2f Hz".format(rates_in))
        print("Simulation time: {}".format(sim_time))


    
    def record_data(self):

        sp_detector_data = nest.GetStatus(self.sp_detector, keys="events")[0]
        sp_times = sp_detector_data["times"]
        sp_senders = sp_detector_data["senders"]
        self.spikes_trains = np.histogram2d(sp_senders, sp_times, bins=[range(1, self.N_neurons+2), np.arange(0, self.sim_params['sim_length']+1, 1)])[0]
        print(self.spikes_trains.shape)

        '''
        self.multimeter_data_E = nest.GetStatus(self.multimeter_network_E, keys="events")[0]
        self.multimeter_data_I = nest.GetStatus(self.multimeter_network_I, keys="events")[0]

        self.multimeter_times = self.multimeter_data_E["times"]
        self.multimeter_senders = self.multimeter_data_E["senders"]
        self.multimeter_V_m_E = self.multimeter_data_E["V_m"]
        self.multimeter_V_m_I = self.multimeter_data_I["V_m"]
        '''
        conn_tuple = nest.GetConnections(source= self.nodes_ex + self.nodes_in, target=self.nodes_ex + self.nodes_in)
        self.adj_matrix = np.zeros((self.N_neurons, self.N_neurons))
        sources = nest.GetStatus(conn_tuple, 'source')
        targets = nest.GetStatus(conn_tuple, 'target')
        weights = nest.GetStatus(conn_tuple, 'weight')

        counter = 0
        for i, j in zip(sources, targets):
            self.adj_matrix[i-1][j-1] = weights[counter]
            counter += 1

        return self.spikes_trains, self.adj_matrix
    
    def record_data_incrementally(self, batch_size=1000):
        sp_detector_data = nest.GetStatus(self.sp_detector, keys="events")[0]
        sp_times = sp_detector_data["times"]
        sp_senders = sp_detector_data["senders"]

        # Define histogram bins based on neuron count and time resolution
        histogram_bins = [range(1, self.N_neurons + 2), np.arange(0, self.sim_params['sim_length'] + 1, 1)]
        spike_histogram = np.zeros((self.N_neurons, len(histogram_bins[1]) - 1))

        # Define file for saving incremental data
        name_spikes = "spikes_incremental.hdf5"
        with h5py.File(name_spikes, 'w') as f:
            spikes_dset = f.create_dataset(
                "spikes", (0, 2), maxshape=(None, 2), dtype='float32'
            )

            # Process data in batches and update the histogram
            for start in range(0, len(sp_times), batch_size):
                end = min(start + batch_size, len(sp_times))
                batch_senders = sp_senders[start:end]
                batch_times = sp_times[start:end]

                # Update histogram incrementally with current batch
                hist_batch, _, _ = np.histogram2d(batch_senders, batch_times, bins=histogram_bins)
                spike_histogram += hist_batch  # Incrementally add to the histogram

                # Save the batch spike data in raw form as well
                batch_data = np.column_stack((batch_senders, batch_times))
                spikes_dset.resize(spikes_dset.shape[0] + batch_data.shape[0], axis=0)
                spikes_dset[-batch_data.shape[0]:] = batch_data

        # Save the final spike histogram as a separate dataset in the HDF5 file
        with h5py.File(name_spikes, 'a') as f:
            f.create_dataset('spike_histogram', data=spike_histogram)

        print(f"Spike data saved incrementally to {name_spikes}")
        return name_spikes

    

    def save_data(self):
        
        name_time = f"{self.sim_params['sim_length']:.1e}".replace('+', '').replace('.', '')
        name_prefix = f"N{self.N_neurons}-T" + name_time
        name_spikes = "spikes-"+name_prefix+".h5"
        name_adj = "connectivity-"+name_prefix+".npy"

        with h5py.File(name_spikes, 'w') as f:
            f.create_dataset('spikes_trains', data=self.spikes_trains)
            f.close()

        np.save(name_adj, self.adj_matrix)

        return name_spikes, name_adj

    def raster(self):
        nest.raster_plot.from_device(self.raster_exc, hist=True, hist_binwidth=.5)
        nest.raster_plot.from_device(self.raster_inh, hist=True, hist_binwidth=.5)
        plt.show()

    #def reset(self):
     #   nest.ResetKernel()
