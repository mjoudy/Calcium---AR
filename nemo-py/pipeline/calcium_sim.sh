#!/bin/bash
#MOAB -N ARRAYJOB
#MOAB -t 0-99
#MOAB -l walltime=1:00:00
#MOAB -l mem=30gb
#MOAB -l nodes=1:ppn=20

echo "STARTED $(date)"
cd /work/ws/nemo/fr_mj200-lasso_reg-0/pipeline
#cd $MOAB_SUBMITDIR
echo "The directory of submission is: $MOAB_SUBMITDIR"
echo "$(pwd)"

echo "Loading modules"
module load devel/python/3.6.9
module load mpi/openmpi/4.0-gnu-9.2

export WORK='/work/ws/nemo/fr_mj200-lasso_reg-0'
export SUBMITDIR=$(pwd)

PYTHON_SCRIPT="python calcium_sim.py"

INPUT_FILE=source_data/t-60e6/chunked/spikes-60e6-ms_$MOAB_JOBARRAYINDEX.npy

$PYTHON_SCRIPT $INPUT_FILE

# !!!!!!! check the number of jobs in the array.!!!!
if [ $MOAB_JOBARRAYINDEX -eq 99 ]; then
    python -c "from loging_pipeline import log_process; log_process('calcium simulation of chunks are done.')"
fi
