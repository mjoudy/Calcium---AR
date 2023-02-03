#!/bin/bash
#MOAB -N nemo_test_2
#MOAB -l walltime=01:40:00
#MOAB -l mem=120gb
#MOAB -l nodes=1:ppn=20

echo "STARTED $(date)"
#cd $MOAB_SUBMITDIR
cd /work/ws/nemo/fr_mj200-conda-0

echo "The directory of submission is: $MOAB_SUBMITDIR"
CurrDir=$MOAB_SUBMITDIR
LDir=`echo $Currdir | awk -F/ '{print $(NF)}'`
SaveDir=$(ws_find conda)/$LDir
echo "Saving Directory"
echo $SaveDir

echo "Loading modules"
module load devel/python/3.6.9
module load mpi/openmpi/4.0-gnu-9.2


echo 'The TMPDIR address is: '
echo $TMPDIR
echo 'The address for Work directory for data transfer from TMPDIR: '
export WORK='/work/ws/nemo/fr_mj200-conda-0'
export SUBMITDIR=$(pwd)
if [ -d "$SaveDir" ];
then

    echo "Save directory already exist"

else
	
	mkdir $SaveDir
fi

python nemo_test.py

#mv $TMPDIR/results	$SaveDir


echo "FINISHED $(date)"
