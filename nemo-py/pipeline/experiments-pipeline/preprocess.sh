#!/bin/bash
#PBS -N ARRAYJOB
#PBS -t 0-99
#PBS -l walltime=00:15:00
#PBS -l mem=30gb
#PBS -l nodes=1:ppn=20

echo "STARTED $(date)"
cd "$PBS_O_WORKDIR"
echo "The directory of submission is: $PBS_O_WORKDIR"
echo "$(pwd)"

echo "Loading modules"
module load devel/python/3.6.9

PYTHON_SCRIPT="python preprocess.py"
export CAL_CHUNK_NAME=$(cat cal_chunk_name.txt)
echo "CAL_CHUNK_NAME: '${CAL_CHUNK_NAME}'"

# Use PBS_ARRAYID for TORQUE job array index
INPUT_FILE="$PBS_O_WORKDIR/chunked-calcium/calcium_tau100-60e6-ms_$PBS_ARRAYID.npy"

$PYTHON_SCRIPT $INPUT_FILE

# Check the number of jobs in the array.
if [ $PBS_ARRAYID -eq 99 ]; then
    python -c "from loging_pipeline import log_process; log_process('calcium signals of chunks are pre-processed.')"
fi
