#!/bin/bash
#PBS -N ARRAYJOB
#PBS -t 0-99
#PBS -l walltime=00:05:00
#PBS -l mem=30gb
#PBS -l nodes=1:ppn=20

echo "STARTED $(date)"
cd "$PBS_O_WORKDIR"
echo "The directory of submission is: $PBS_O_WORKDIR"
echo "Current working directory: $(pwd)"

echo "Loading modules"
module load devel/python/3.6.9

PYTHON_SCRIPT="python calcium_sim.py"
export CHUNK_NAME=$(cat chunk_name.txt)
echo "CHUNK_NAME: '${CHUNK_NAME}'"

INPUT_FILE="${PBS_O_WORKDIR}/chunked/${CHUNK_NAME}_${PBS_ARRAYID}.npy"
echo "INPUT_FILE: '${INPUT_FILE}'"

$PYTHON_SCRIPT "$INPUT_FILE"

# Check the number of jobs in the array.
if [ $PBS_ARRAYID -eq 99 ]; then
    python -c "from logging_pipeline import log_process; log_process('Calcium simulation of chunks is done.')"
fi
