import nest
import nest.raster_plot

import matplotlib.pyplot as plt
import numpy as np

import time
from numpy import exp

import csv
#import pandas as pd
import h5py

#nest.ResetKernel()

#startbuild = time.time()
#nest.SetKernelStatus({"local_num_threads": 20})

##dt = 0.1  # the resolution in ms
##simtime = 10000 # Simulation time in ms

##g = 6.0  # ratio inhibitory weight/excitatory weight
##eta = 2.0  # external rate relative to threshold rate
##epsilon = 0.1  # connection probability

##order = 250
##NE = 4 * order  # number of excitatory neurons
##NI = 1 * order  # number of inhibitory neurons
##N_neurons = NE + NI  # number of neurons in total
N_rec = 50  # record from 50 neurons

##CE = int(epsilon * NE)  # number of excitatory synapses per neuron
##CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
##C_tot = int(CI + CE)  # total number of synapses per neuron

##delay = 1.5  # synaptic delay in ms
##tauMem = 20.0  # time constant of membrane potential in ms
##theta = 20.0  # membrane threshold potential in mV
##neuron_params = {"C_m": 1.0,
                 "tau_m": tauMem,
                 "t_ref": 2.0,
                 "E_L": 0.0,
                 "V_reset": 0.0,
                 "V_m": 0.0,
                 "V_th": theta}
##J = 8.  # postsynaptic amplitude in mV
##J_ex = J  # amplitude of excitatory postsynaptic potential
##J_in = -g * J_ex  # amplitude of inhibitory postsynaptic potential

##nu_th = theta / (J * CE * tauMem)
##nu_ex = eta * nu_th
##p_rate = 10000.0 * nu_ex * CE

##nest.SetKernelStatus({"resolution": dt, "print_time": True,
##                      "overwrite_files": True})

print("Building network")

##nest.SetDefaults("iaf_psc_delta", neuron_params)
##nest.SetDefaults("poisson_generator", {"rate": p_rate})

#DC generator
##nest.SetDefaults("dc_generator", {"amplitude": 6.})

##nodes_ex = nest.Create("iaf_psc_delta", NE, params=neuron_params)
##nodes_in = nest.Create("iaf_psc_delta", NI, params=neuron_params)

#DC Generator
##noise = nest.Create("poisson_generator")
##espikes = nest.Create("spike_recorder")
##ispikes = nest.Create("spike_recorder")

#neuron test
neuron_test = nest.Create("iaf_psc_delta", 1)

##multimeter_network_E = nest.Create("multimeter")
##nest.SetStatus(multimeter_network_E, {"record_from":["V_m"]})

##multimeter_network_I = nest.Create("multimeter")
##nest.SetStatus(multimeter_network_I, {"record_from":["V_m"]})

##sp_detector = nest.Create("spike_recorder")

###### !!!!!!! ########

nest.CopyModel("static_synapse", "excitatory",
               {"weight": J_ex, "delay": delay})
nest.CopyModel("static_synapse", "inhibitory",
               {"weight": J_in, "delay": delay})

##nest.Connect(noise, nodes_ex, syn_spec="excitatory")
##nest.Connect(noise, nodes_in, syn_spec="excitatory")

#neuron test noise
nest.Connect(noise, neuron_test, syn_spec="excitatory")

##nest.Connect(multimeter_network_E, nodes_ex)
##nest.Connect(multimeter_network_I, nodes_in)

##nest.Connect(nodes_ex + nodes_in, sp_detector, syn_spec="excitatory")

##nest.Connect(nodes_ex[:N_rec], espikes, syn_spec="excitatory")
##nest.Connect(nodes_in[:N_rec], ispikes, syn_spec="excitatory")

print("Connecting network")
print("Excitatory connections")

##conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}
##nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, "excitatory")

print("Inhibitory connections")

##conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
##nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, "inhibitory")

endbuild = time.time()
print("Simulating")

nest.Simulate(simtime)

endsimulate = time.time()

events_ex = nest.GetStatus(espikes, "n_events")[0]
events_in = nest.GetStatus(ispikes, "n_events")[0]

rate_ex = events_ex / simtime * 1000.0 / N_rec
rate_in = events_in / simtime * 1000.0 / N_rec

num_synapses = (nest.GetDefaults("excitatory")["num_connections"] +
                nest.GetDefaults("inhibitory")["num_connections"])

build_time = endbuild - startbuild
sim_time = endsimulate - endbuild

print("Brunel network simulation (Python)")
print("Number of neurons : {0}".format(N_neurons))
print("Number of synapses: {0}".format(num_synapses))
print("       Exitatory  : {0}".format(int(CE * N_neurons) + N_neurons))
print("       Inhibitory : {0}".format(int(CI * N_neurons)))
print("Excitatory rate   : %.2f Hz" % rate_ex)
print("Inhibitory rate   : %.2f Hz" % rate_in)
print("Building time     : %.2f s" % build_time)
print("Simulation time   : %.2f s" % sim_time)

sp_detector_data = nest.GetStatus(sp_detector, "events")[0]
sp_times = sp_detector_data["times"]
sp_senders = sp_detector_data["senders"]

#Get Connectivity matrix
conn_tuple = nest.GetConnections(source= nodes_ex+nodes_in, target=nodes_ex+nodes_in)

conn_matrix = np.zeros((N_neurons, N_neurons))

sources = nest.GetStatus(conn_tuple, 'source')
targets = nest.GetStatus(conn_tuple, 'target')
weights = nest.GetStatus(conn_tuple, 'weight')

counter = 0

for i, j in zip(sources, targets):
    conn_matrix[j-1, i-1] = weights[counter]
    counter+=1
    
np.save("connectivity-60e6-ms", conn_matrix)

spikes = np.histogram2d(sp_senders, sp_times, bins=[range(1, 1252), np.arange(0, simtime+1, 1)])[0]

#np.save("spikes-60e6-ms", spikes)
#np.shape(spikes)

with h5py.File('your_array_no_chunks.hdf5', 'w') as f:
    f.create_dataset('large_array', data=spikes)