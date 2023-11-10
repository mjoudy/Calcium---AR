#!/bin/bash
#MSUB -N ARRAYJOB
#MSUB -t 0-99
#MSUB -l walltime=1:00:00
#MSUB -l mem=20gb
#MSUB -l nodes=100:ppn=20

module load devel/python/3.6.9
module load mpi/openmpi/4.0-gnu-9.2

PYTHON_SCRIPT=python preprocess.py

INPUT_FILE=./spikes-10e4-ms_$MOAB_JOBARRAYINDEX.npy

$PYTHON_SCRIPT $INPUT_FILE
