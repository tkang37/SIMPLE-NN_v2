#!/bin/sh
#PBS -l nodes=1:g3:ppn=20
#PBS -N Test

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE > nodefile
NPROC=`wc -l < $PBS_NODEFILE`


source activate pytorch
python test_datagenerator.py > output.dat
