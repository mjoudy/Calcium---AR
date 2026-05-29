import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig

from pyspark import SparkContext, SparkConf
import pyspark


def sim_calcium(spikes, tau=100, neuron_id=500):

    if neuron_id == -1:
        N = np.shape(spikes)[0]
        wup_time = 1000
        spikes = spikes[:, wup_time:]
        sim_dur = np.shape(spikes)[1]

        noise_intra = np.random.normal(0, 0.01, (N, sim_dur))
        spikes_noisy = spikes + noise_intra

        calcium = np.zeros((N, sim_dur))
        calcium_nsp = np.zeros((N, sim_dur))
        dt = 1
        const_A = np.exp((-1/tau)*dt)

        calcium[:, 0] = spikes[:, 0]
        calcium_nsp[:, 0] = spikes[:, 0]

        for t in range(1, sim_dur):
            calcium[:, t] = const_A*calcium[:, t-1] + spikes[:, t]

        for t in range(1, sim_dur):
            calcium_nsp[:, t] = const_A*calcium_nsp[:, t-1] + spikes_noisy[:, t]

        noise_recording = np.random.normal(0,1, (N, sim_dur))
        calcium_noisy = calcium + noise_recording
        calcium_nsp_noisy = calcium_nsp + noise_recording
    else:
        wup_time = 1000
        spikes = spikes[neuron_id, wup_time:]
        sim_dur = np.shape(spikes)[0]

        noise_intra = np.random.normal(0, 0.01, sim_dur)
        spikes_noisy = spikes + noise_intra

        calcium = np.zeros(sim_dur)
        calcium_nsp = np.zeros(sim_dur)
        dt = 1
        const_A = np.exp((-1/tau)*dt)

        calcium[0] = spikes[0]
        calcium_nsp[0] = spikes[0]

        for t in range(1, sim_dur):
            calcium[t] = const_A*calcium[t-1] + spikes[t]

        for t in range(1, sim_dur):
            calcium_nsp[t] = const_A*calcium_nsp[t-1] + spikes_noisy[t]

        noise_recording = np.random.normal(0,1, sim_dur)
        calcium_noisy = calcium + noise_recording
        calcium_nsp_noisy = calcium_nsp + noise_recording

    #return calcium, calcium_noisy, calcium_nsp, calcium_nsp_noisy
    return calcium_nsp_noisy, spikes


conf = SparkConf().setAppName("SimCalciumApp")
sc = SparkContext(conf=conf)

spikes_add = '/home/fr/fr_fr/fr_mj200/spikes-60e6-ms.npy'
spikes = np.load(spikes_add)

rdd = sc.parallelize(spikes, numSlices=10)
result = rdd.map(sim_calcium(spikes, neuron_id=-1))
final_result = result.collect()

#calcium_signal, spikes = sim_calcium(spikes, neuron_id=-1)
#calcium_signal.shape

final_result.shape
print(final_result.shape)
