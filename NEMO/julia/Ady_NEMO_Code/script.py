## script

import numpy as np
import sys

# sys.path.insert(2,'/home/fr/fr_fr/fr_jg1037/nest-new-sp/lib64/python3.6/site-packages/')
sys.path.insert(2, '/opt/bwhpc/common/conda/envs/nest-2.20.0/lib/python3.7/site-packages/')

import nest

rank = nest.Rank()
par = __import__(sys.argv[1].rpartition("/")[-1].partition(".")[0])
msd = int(sys.argv[2])
direc = sys.argv[1].rpartition('/')[0] + "/data/"

# Random seeds
pyrngs = [np.random.RandomState(s) for s in range(msd, msd + par.total_num_virtual_procs)]
grng_seed = msd + par.total_num_virtual_procs
rng_seeds = range(msd + par.total_num_virtual_procs + 1, msd + 2 * par.total_num_virtual_procs + 1)

np.save(direc + "master_seed.npy", msd)

# Initialize NEST kernel
nest.ResetKernel()
#nest.EnableStructuralPlasticity()
nest.SetKernelStatus({
    'resolution': par.dt,
    'print_time': False,
    #'structural_plasticity_update_interval': int(par.MSP_update_interval / par.dt),
    # update interval for MSP in time steps
    'total_num_virtual_procs': par.total_num_virtual_procs,
    'grng_seed': grng_seed,
    'rng_seeds': rng_seeds,
})

nest.SetDefaults(par.neuron_model, par.neuron_params)

# Create generic neuron with Axon and Dendrite
nest.CopyModel(par.neuron_model, 'excitatory')
nest.CopyModel(par.neuron_model, 'inhibitory')

# growth curves
# gc_den = {'growth_curve': par.growth_curve, 'z': par.z0, 'growth_rate': -par.slope * par.eps, 'eps': par.eps,
#           'continuous': False, 'tau_vacant': par.tau_vacant}
# gc_axon = {'growth_curve': par.growth_curve, 'z': par.z0, 'growth_rate': -par.slope * par.eps, 'eps': par.eps,
#            'continuous': False, 'tau_vacant': par.tau_vacant}

#nest.SetDefaults('excitatory', 'synaptic_elements', {'Axon_exc': gc_axon, 'Den_exc': gc_den})

# Create synapse models
nest.CopyModel(par.synapse_model, 'msp_excitatory')
nest.SetDefaults('msp_excitatory', {'weight': par.J, 'delay': par.delay})
nest.CopyModel("static_synapse", "excitatory_static", {"weight": par.J, "delay": par.delay})
nest.CopyModel("static_synapse", "inhibitory_static", {"weight": -par.g * par.J, "delay": par.delay})
nest.CopyModel("static_synapse", "device", {"weight": par.J, "delay": par.delay})

# Use SetKernelStatus to activate the synapse model
# nest.SetKernelStatus({
#    'structural_plasticity_synapses': { 'syn1': {  'model': 'msp_excitatory',  'post_synaptic_element': 'Den_exc',  'pre_synaptic_element': 'Axon_exc', }  },
#    'autapses': False,
# })

# Create nodes
pop_exc = nest.Create('excitatory', par.NE)
pop_inh = nest.Create('inhibitory', par.NI)
pop_sample = nest.Create('excitatory', par.N_sample, params={'V_th': 1e6})  ## <------
spike_detector = nest.Create("spike_detector")
external_input = nest.Create('poisson_generator')
nest.SetStatus(external_input, {"rate": par.rate})
VM_sample = nest.Create('multimeter', 1, params={'withtime': True, 'record_from': ['V_m']})
SD_sample = nest.Create('spike_detector')

# Connect nodes
nest.Connect(pop_exc, pop_inh + pop_exc, {'rule': 'fixed_indegree', 'indegree': par.CE}, 'excitatory_static') ### <------ Changes for Static Brunel
nest.Connect(pop_exc, pop_sample, conn_spec={'rule': 'fixed_indegree', 'indegree': par.CE}, syn_spec='excitatory_static')  ## <------
nest.Connect(pop_inh, pop_sample, conn_spec={'rule': 'fixed_indegree', 'indegree': par.CI}, syn_spec='inhibitory_static')  ## <------
nest.Connect(pop_inh, pop_exc + pop_inh, {'rule': 'fixed_indegree', 'indegree': par.CI}, 'inhibitory_static')
nest.Connect(external_input, pop_exc + pop_inh + pop_sample, 'all_to_all', syn_spec="device")  ## <---------
nest.Connect(pop_exc + pop_inh, spike_detector, 'all_to_all', syn_spec="device")
nest.Connect(pop_sample, SD_sample)  ## <-----
nest.Connect(VM_sample, pop_sample)  ## <-----

local_connections = nest.GetConnections(pop_exc + pop_inh, pop_exc + pop_inh)
sources = nest.GetStatus(local_connections, 'source')
targets = nest.GetStatus(local_connections, 'target')
weights = nest.GetStatus(local_connections, 'weight')

extension = "0.0_" + str(rank) + ".npy"

np.save(direc + "sources_" + extension, sources)
np.save(direc + "targets_" + extension, targets)
np.save(direc + "weights_" + extension, weights)


def simulate_cicle(growth_steps):
    growth_step = growth_steps[1] - growth_steps[0]
    for simulation_time in growth_steps:
        nest.SetStatus(spike_detector,
                       {'start': simulation_time + growth_step - 20000., 'stop': simulation_time + growth_step})
        nest.SetStatus(VM_sample,
                       {'start': simulation_time + growth_step + 1., 'stop': simulation_time + growth_step + 1. + 3e3})
        nest.SetStatus(SD_sample,
                       {'start': simulation_time + growth_step + 1., 'stop': simulation_time + growth_step + 1. + 3e3})

        nest.Simulate(growth_step)
#        nest.DisableStructuralPlasticity()  ### <-------- 
        nest.Simulate(3e3)  ## <------- simulate the sample population for 3 seconds
 #       nest.EnableStructuralPlasticity()  ### <--------

        local_connections = nest.GetConnections(pop_exc+pop_inh, pop_exc+pop_inh) 
        sources = nest.GetStatus(local_connections, 'source')
        targets = nest.GetStatus(local_connections, 'target')
        weights = nest.GetStatus(local_connections, 'weight')

        extension = str(simulation_time + growth_step) + "_" + str(rank) + ".npy"
        extension_sample = str(simulation_time + growth_step) + "_sample_" + str(rank) + ".npy"  ## <-------

        np.save(direc + "sources_" + extension, sources)
        np.save(direc + "targets_" + extension, targets)
        np.save(direc + "weights_" + extension, weights)
        del local_connections

        events = nest.GetStatus(spike_detector, 'events')[0]
        times = events['times']
        senders = events['senders']

        np.save(direc + "times_" + extension, times)
        np.save(direc + "senders_" + extension, senders)
        nest.SetStatus(spike_detector, 'n_events', 0)

        ## ------> <-------
        events_sample = nest.GetStatus(SD_sample, 'events')[0]
        SD_sample_times = events_sample['times']
        SD_sample_senders = events_sample['senders']

        np.save(direc + "SD_sample_times" + extension_sample, SD_sample_times)
        np.save(direc + "SD_sample_senders" + extension_sample, SD_sample_senders)

        VM_sample_data = nest.GetStatus(VM_sample, 'events')[0]
        VM_sample_V_m = VM_sample_data['V_m']
        VM_sample_times = VM_sample_data['times']

        np.save(direc + "VM_sample_V_m" + extension_sample, VM_sample_V_m)
        np.save(direc + "VM_sample_times" + extension_sample, VM_sample_times)

        nest.SetStatus(SD_sample, 'n_events', 0)
        nest.SetStatus(VM_sample, 'n_events', 0)


# Grow network
growth_steps = np.arange(0, par.growth_time, par.pre_step)
simulate_cicle(growth_steps)


