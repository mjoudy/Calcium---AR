#!/bin/bash
#PBS -N conn_inf
#PBS -l walltime=00:05:00
#PBS -l mem=120gb
#PBS -l nodes=1:ppn=20

echo "STARTED $(date)"
cd "$PBS_O_WORKDIR"

echo "The directory of submission is: $PBS_O_WORKDIR"
echo "Loading modules"
module load devel/python/3.6.9

echo 'The TMPDIR address is: '
echo $TMPDIR

python conn_inf.py

echo "FINISHED $(date)"
