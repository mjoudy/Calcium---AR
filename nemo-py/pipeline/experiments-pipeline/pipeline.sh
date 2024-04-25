#!/bin/bash

# Stage 1: chunk_data.sh
jobid1=$(qsub chunk_data.sh)
echo "Submitted chunk_data.sh, job ID: $jobid1"

# Stage 2: calcium_sim.sh (Array Job)
# Assuming the array tasks are indexed from 0-99 as an example
jobid2=$(qsub -W depend=afterok:$jobid1 calcium_sim.sh)
echo "Submitted calcium_sim.sh (Array Job), job ID: $jobid2"

# Stage 3: preprocess.sh (Array Job)
# Assuming the array tasks are indexed from 0-99 as an example
jobid3=$(qsub -W depend=afterokarray:$jobid2 preprocess.sh)
echo "Submitted preprocess.sh (Array Job), job ID: $jobid3"

# Stage 4: concat_chunks.sh
jobid4=$(qsub -W depend=afterokarray:$jobid3 concat_chunks.sh)
echo "Submitted concat_chunks.sh, job ID: $jobid4"

# Stage 5: conn_inf.sh
jobid5=$(qsub -W depend=afterok:$jobid4 conn_inf.sh)
echo "Submitted conn_inf.sh, job ID: $jobid5"
