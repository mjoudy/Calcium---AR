#!/bin/bash
#PBS -N chunk_data
#PBS -l walltime=00:05:00
#PBS -l mem=120gb
#PBS -l nodes=1:ppn=20

cd "$PBS_O_WORKDIR"

echo "STARTED $(date)"
echo "The directory of submission is: $PBS_O_WORKDIR"
echo "Loading modules"
module load devel/python/3.6.9

echo 'The TMPDIR address is: '
echo $TMPDIR

python chunk_data.py

echo "FINISHED $(date)"
