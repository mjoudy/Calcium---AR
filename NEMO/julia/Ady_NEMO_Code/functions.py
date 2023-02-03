# functions

import numpy as np
import scipy.sparse as sp

def gather(label,it_range,form):
    '''
    Concatenates files from different ranks/steps into one single array
    Inputs are:
    label     - string with the main part of the name of the file
    it_range  - range with the varying part of the file's name
    form      - string with the format of the file (ex: 'npy')
    * name of the file should be in the format: label_it.format
    Returns an array with the values from all ranks
    '''
    array = np.array([])
    for it in it_range:
        array = np.hstack((array,np.load(label+"_"+str(it)+"."+form)))
    return array

def construct_matrix_st(direc,N,step,n_rank):
    '''
    Creates connectivity matrix based on files sources.npy and targets.npy created during the simulation
    Inputs are:
    N         - total number of neurons in the connectivity matrix (int)
    step      - step of the simulation to create the connectivity matrix
    n_rank    - number of ranks used for the simulation (int)
    * sources (targets) files should contain a list of sources (targets) of each connection
    * neuron tags should go from 1-N
    * sources (targets) files should have a name in the format: sources_step_rank.npy
    Returns connectivity matrix in the format csr (from scipy.sparse)
    '''
    sources = gather(direc+"data/sources_"+str(step),np.arange(n_rank),"npy")
    targets = gather(direc+"data/targets_"+str(step),np.arange(n_rank),"npy")
    weights = gather(direc+"data/weights_"+str(step),np.arange(n_rank),"npy")

#    weights = np.ones(len(weights))

    sources = sources[targets<=N]
    targets = targets[targets<=N]
    targets = targets[sources<=N]
    sources = sources[sources<=N]

    sources -= 1
    targets -= 1

    matrix_final = sp.coo_matrix((weights,(targets,sources)),shape=(N,N)).tocsr()
    return matrix_final

def save_delta_connectivity(direc,N,matrix_initial,sim_steps,n_rank):
    '''
    Saves delta_connectivity files (only changes in the connectivity matrix from successive time steps) from sources and targets files
    Inputs are:
    N                 - total number of neurons in the connectivity matrix (int)
    matrix_initial    - connectivity matrix before initial step on sim_steps (empty matrix if starting from scratch) on format csr (from scipy.sparse)
    sim_steps         - range with the steps of the simulation to create delta_connectivity files
    n_rank            - number of ranks used for the simulation (int)
    * sources (targets) files should contain a list of sources (targets) of each connection
    * neuron tags should go from 1-N
    * sources (targets) files should have a name in the format: sources_step_rank.npy
    Saves files with name in format: delta_connectivity_step.npy in the format csr (from scipy.sparse)
    '''
    for step in sim_steps:
        matrix_final = construct_matrix_st(direc,N,step,n_rank)
        delta_connectivity = matrix_final - matrix_initial
        np.save(direc+"data/delta_connectivity_"+str(float(step))+".npy",delta_connectivity)
        matrix_initial = matrix_final*1

def update_matrix_hdf5(N,matrix_initial,data,indices,indptr):
    return matrix_initial + sp.csr_matrix((data,indices,indptr),shape=(N,N))

def construct_matrix_hdf5(N,data_file,matrix_initial,sim_steps):
    matrix = matrix_initial
    for step in sim_steps:
        data_step = data_file[str(step)]
        data = data_step['delta_data']
        indices = data_step['delta_indices']
        indptr = data_step['delta_indptr']
        matrix = update_matrix_hdf5(N,matrix,data,indices,indptr)
    return matrix

def construct_matrix_dc(matrix,sim_steps):
    '''
    Constructs connectivity matrix from delta_connectivity files
    Inputs are:
    matrix    - initial matrix (empty matrix if starting from scratch) in format csr (from scipy.sparse)
    sim_steps - range of simulation steps over which matrix should be constructed
    Returns matrix in format csr (from scipy.sparse) starting with input matrix after steps on sim_steps
    '''
    for step in sim_steps:
        delta_connectivity = np.load("data/delta_connectivity_"+str(step)+".npy")
        delta_connectivity = delta_connectivity[()]
        matrix = matrix + delta_connectivity
    return matrix


""" Ady does shit """

def AdjacencyMatrix(direc, N, step, n_rank):
    Adj_matrix = np.zeros([N, N])

    global src, tgt
    src = gather(direc+"data/sources_"+str(step),np.arange(n_rank),"npy")
    tgt = gather(direc+"data/targets_"+str(step),np.arange(n_rank),"npy")
    # weights = gather(direc+"data/weights_"+str(step),np.arange(n_rank),"npy")

    # return sources, targets
    # print("src:"+str(np.min(sources))+ str(np.max(sources)))
    # print("tgt:"+str(np.min(targets))+ str(np.max(targets)))
    #
    # for _ in range(len(sources)):
    #     Adj_matrix[sources[_]-1][targets[_]-1] += 1
    #
    # return Adj_matrix

