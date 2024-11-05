#!/bin/bash
#MOAB -N lasso
#MOAB -l walltime=01:40:00
#MOAB -l mem=240gb
#MOAB -l nodes=1:ppn=20

echo "STARTED $(date)"
#cd $MOAB_SUBMITDIR
cd /home/fr/fr_fr/fr_mj200

echo "The directory of submission is: $MOAB_SUBMITDIR"
CurrDir=$MOAB_SUBMITDIR
LDir=`echo $Currdir | awk -F/ '{print $(NF)}'`
SaveDir=/home/fr/fr_fr/fr_mj200/$LDir
echo "Saving Directory"
echo $SaveDir

echo "Loading modules"
module load devel/python/3.6.9
module load mpi/openmpi/4.0-gnu-9.2


echo 'The TMPDIR address is: '
echo $TMPDIR
echo 'The address for Work directory for data transfer from TMPDIR: '
export WORK='/home/fr/fr_fr/fr_mj200'
export SUBMITDIR=$(pwd)
if [ -d "$SaveDir" ];
then

    echo "Save directory already exist"

else

        mkdir $SaveDir
fi

python lasso.py

#mv $TMPDIR/results     $SaveDir


echo "FINISHED $(date)"
~                           