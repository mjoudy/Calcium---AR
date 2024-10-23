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

        self.nu_th = 




    def setup_neurons(self):

        nest.setDefaults("")