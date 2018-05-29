#!/bin/bash

#SBATCH
#SBATCH --mail-type=end
#SBATCH --mail-user=ali39@jhu.edu
#SBATCH --output=_out/%A.out 
#SBATCH --error=_out/%A.err

# Author: Adam Li (ali39@jhu.edu).
# Created on 2017-10-31. 
#---------------------------------------------------------------------
# SLURM job script to run Singularity Keras/Tflow Training
#---------------------------------------------------------------------
############################# LOAD IN MODULES AND UNLOAD #############
module unload git
ml anaconda-python/3.6
ml python/3.6.5
ml cuda/9.0
ml singularity 

# activate our conda environment
source activate dnn

# cd to the scratch location 
cd /scratch/users/$USER/dnn-unsupervised/bin_main/

########################## DEBUG OUTPUT  ###########################
# grep for SLURM_EXPORT_ENV
echo "log data dir is : ${logdatadir}"
echo "Output data dir is: ${outputdatadir}"
echo "Train data dir is: ${traindatadir}"
echo "Test data dir is: ${testdatadir}"
echo "Patient to analyze is: ${patient}"
echo ${CUDA_VISIBLE_DEVICES}

nvidia-smi

################### SETUP SINGULARITY CONTAINER  ####################
# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER

echo $PWD
echo "Running training model"
# docker://python:latest
# singularity options: 
# -vvv -d = very verbose / debugging mode
# -B = bind path /scratch /work /data

# for training fragnet
echo "This is the command being run: singularity exec -B /scratch/ --nv ../pytorch.simg python ./trainfragaux/main_refactored.py \
${outputdatadir} ${logdatadir} ${traindatadir} ${datadir}"

singularity exec -B /scratch/ --nv ../pytorch.simg python ./slurm/exp001/main.py ${outputdatadir} ${logdatadir} ${traindatadir} ${testdatadir} ${patient}


exit