## Params

# Parameter of the simulation
total_num_virtual_procs = 20*2 #since  ppn=20 | Julia uses = 40
dt                      = 0.1
MSP_update_interval     = 100                                   # update interval for MSP in ms
growth_time             = 750000.                                # simulation time in ms
cicles                  = 10
pre_step                = growth_time/cicles
delay                   = 1.5


# Parameters for asynchronous irregular firing
g       = 75.0
eta     = 1.5
epsilon = 0.1  # connection probability

order     = 2500
NE        = 4*order
NI        = 1*order
N_neurons = NE+NI
N_sample = 50

CE = int(epsilon*NE)                                              # number of excitatory synapses per neuron
CI = int(epsilon*NI)                                              # number of inhibitory synapses per neuron  


# Initialize the parameters of the integrate and fire neuron
neuron_model    = "iaf_psc_delta"
CMem            = 250.0
tauMem          = 20.0
theta           = 20.0
tau_Ca          = 10000.
beta_Ca         = 1./tau_Ca
J               = 0.1                                           # postsynaptic amplitude in mV

neuron_params   = {
                    "C_m"       : CMem,
                    "tau_m"     : tauMem,
                    "t_ref"     : 2.0,
                    "E_L"       : 0.0,
                    "V_reset"   : 10.0,
                    "V_m"       : 0.0,
                    "beta_Ca"   : beta_Ca,
                    "tau_Ca"    : tau_Ca,
                    "V_th"      : theta
                   }


# threshold rate, equivalent rate of events needed to
# have mean input current equal to threshold
nu_th  = theta/(J*CE*tauMem)
nu_ex  = eta*nu_th
rate = 1000.0*nu_ex*CE  

 
# Parameter for synpatic elements' growth curve
growth_curve = "linear"
z0           = 0.
slope        = -0.5
eps          = 0.008
tau_vacant   = 1e-12

# Parameter for structural plasticity synapse model
synapse_model   = "static_synapse"

