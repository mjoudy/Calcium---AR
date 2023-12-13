#!/bin/bash
#MOAB -N ARRAYJOB
#MOAB -t 0-99
#MOAB -l walltime=1:00:00
#MOAB -l mem=20gb
#MOAB -l nodes=100:ppn=20

module load devel/python/3.6.9
module load mpi/openmpi/4.0-gnu-9.2

PYTHON_SCRIPT=python preprocess.py

INPUT_FILE=./spikes-10e4-ms_$MOAB_JOBARRAYINDEX.npy

$PYTHON_SCRIPT $INPUT_FILE
