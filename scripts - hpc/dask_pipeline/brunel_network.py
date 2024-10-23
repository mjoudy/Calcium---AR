import nest
import nest.raster_plot
import matplotlib.pyplot as plt
import numpy as np
import time
import h5py
import json

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

        self.strat_setup = None
        self.end_setup = None

    
    def ext_input(self):

        nu_th = self.neuron_params['V_th'] / (self.J_ex * self.CE * self.neuron_params['tau_m'])
        nu_ex = self.network_params['eta'] * self.nu_th
        p_rate = 10000.0 * self.nu_ex * self.CE

        return p_rate
    

    def setup_kernel(self):

        nest.ResetKernel()
        self.strat_setup = time.time()
        nest.SetKernelStatus({"local_num_threads": self.sim_params['cpu_num']})
        nest.SetKernelStatus({"resolution": self.sim_params['time_res'], "print_time": True,
                      "overwrite_files": True})


    def setup_elements(self):
        
        nest.SetDefaults("iaf_psc_delta", self.neuron_params)
        nest.SetDefaults("poisson_generator", {"rate": self.ext_input()})
        nest.SetDefaults("dc_generator", {"amplitude": 6.})

        self.nodes_ex = nest.Create("iaf_psc_delta", self.NE, params=self.neuron_params)
        self.nodes_in = nest.Create("iaf_psc_delta", self.NI, params=self.neuron_params)

        self.noise = nest.Create("poisson_generator")
        self.espikes = nest.Create("spike_recorder")
        self.ispikes = nest.Create("spike_recorder")
        '''
        multimeter_network_E = nest.Create("multimeter")
        nest.SetStatus(multimeter_network_E, {"record_from":["V_m"]})

        multimeter_network_I = nest.Create("multimeter")
        nest.SetStatus(multimeter_network_I, {"record_from":["V_m"]})
        '''
        self.sp_detector = nest.Create("spike_recorder")

    def connect_elements(self):

        nest.Connect(self.noise, self.nodes_ex, syn_spec="excitatory")
        nest.Connect(self.noise, self.nodes_in, syn_spec="excitatory")

        nest.Connect(self.nodes_ex[:self.network_params['N_rec']], self.espikes)
        nest.Connect(self.nodes_in[:self.network_params['N_rec']], self.ispikes)

        nest.Connect(self.nodes_ex, self.nodes_ex, conn_spec={'rule': 'pairwise_bernoulli', 'p': self.network_params['epsilon']}, syn_spec="excitatory")
        nest.Connect(self.nodes_ex, self.nodes_in, conn_spec={'rule': 'pairwise_bernoulli', 'p': self.network_params['epsilon']}, syn_spec="excitatory")
        nest.Connect(self.nodes_in, self.nodes_ex, conn_spec={'rule': 'pairwise_bernoulli', 'p': self.network_params['epsilon']}, syn_spec="inhibitory")
        nest.Connect(self.nodes_in, self.nodes_in, conn_spec={'rule': 'pairwise_bernoulli', 'p': self.network_params['epsilon']}, syn_spec="inhibitory")

        self.end_setup = time.time()