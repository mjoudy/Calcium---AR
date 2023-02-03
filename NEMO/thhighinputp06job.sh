#!/bin/bash
#MOAB -N thhighinputp06job
#MOAB -l walltime=01:40:00
#MOAB -l mem=122gb
#MOAB -l nodes=1:ppn=20

echo "STARTED $(date)"
#cd $MOAB_SUBMITDIR
cd /work/ws/nemo/fr_am762-conda-0/nestnew

echo "The directory of submission is: $MOAB_SUBMITDIR"
CurrDir=$MOAB_SUBMITDIR
LDir=`echo $Currdir | awk -F/ '{print $(NF)}'`
SaveDir=$(ws_find conda)/$LDir
echo "Saving Directory"
echo $SaveDir

echo "Loading modules"
module load devel/python/3.6.9
module load mpi/openmpi/4.0-gnu-9.2

source /work/ws/nemo/fr_am762-conda-0/nestnew/bin/nest_vars.sh

echo 'The TMPDIR address is: '
echo $TMPDIR
echo 'The address for Work directory for data transfer from TMPDIR: '
export WORK='/work/ws/nemo/fr_am762-conda-0/nestnew/'
export SUBMITDIR=$(pwd)
if [ -d "$SaveDir" ];
then

    echo "Save directory already exist"

else
	
	mkdir $SaveDir
fi

python th_highinput_nemo_p06.py

#mv $TMPDIR/results	$SaveDir


echo "FINISHED $(date)"
