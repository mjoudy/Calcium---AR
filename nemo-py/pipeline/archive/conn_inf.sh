#!/bin/bash
#MOAB -N conn_inf
#MOAB -l walltime=00:55:00
#MOAB -l mem=120gb
#MOAB -l nodes=1:ppn=20

echo "STARTED $(date)"
#cd $MOAB_SUBMITDIR
cd /work/ws/nemo/fr_mj200-lasso_reg-0/pipeline

echo "The directory of submission is: $MOAB_SUBMITDIR"
CurrDir=$MOAB_SUBMITDIR
LDir=`echo $Currdir | awk -F/ '{print $(NF)}'`
SaveDir=$(ws_find pipeline)/$LDir
echo "Saving Directory"
echo $SaveDir

echo "Loading modules"
module load devel/python/3.6.9
module load mpi/openmpi/4.0-gnu-9.2


echo 'The TMPDIR address is: '
echo $TMPDIR
echo 'The address for Work directory for data transfer from TMPDIR: '
export WORK='/work/ws/nemo/fr_mj200-lasso_reg-0'
export SUBMITDIR=$(pwd)
if [ -d "$SaveDir" ];
then

    echo "Save directory already exist"

else

        mkdir $SaveDir
fi

python conn_inf.py

#mv $TMPDIR/results     $SaveDir


echo "FINISHED $(date)"
