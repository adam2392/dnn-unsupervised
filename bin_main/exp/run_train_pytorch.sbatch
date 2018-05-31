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

# POSSIBLE COMMANDS:
# docker://python:latest
# singularity options: 
# -vvv -d = very verbose / debugging mode
# -B = bind path /scratch /work /data

############################# LOAD IN MODULES AND UNLOAD #############
module unload git
# ml anaconda-python/3.6
ml cuda/9.0
ml singularity 

# activate our conda environment
# source activate dnn

# add CUDA paths
# CPATH="/usr/local/cuda/include:$CPATH"
# PATH="/usr/local/cuda/bin:$PATH"
# LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
# CUDA_HOME="/usr/local/cuda"
# export CPATH PATH LD_LIBRARY_PATH CUDA_HOME

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

# print out nvidia's system management interface log
nvidia-smi

################### SETUP SINGULARITY CONTAINER  ####################
# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER
singularity pull --name pytorch.simg shub://marcc-hpc/pytorch

echo $PWD
echo "Running training model"
# echo "This is the command being run: singularity exec -B /scratch/ --nv ../pytorch.simg python ./trainfragaux/main_refactored.py \
# ${outputdatadir} ${logdatadir} ${traindatadir} ${datadir}"

# run training here in this
pycmd="python ./slurm/exp001/main.py ${traindatadir} ${testdatadir} --output_data_dir ${outputdatadir} --log_data_dir ${logdatadir} --patient_to_loo ${patient} --expname ${expname}"
# ${traindatadir} ${testdatadir} --output_data_dir ${outputdatadir} --log_data_dir ${logdatadir} --patient_to_loo ${patient} --expname ${expname}
singularity -vvv -d exec -B /scratch/ --nv ./pytorch.simg python ./exp/exp001/main.py ${traindatadir} ${testdatadir} ${outputdatadir} ${logdatadir} ${patient} ${expname}

exit