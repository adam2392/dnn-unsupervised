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
# module unload git
ml cuda/9.0
ml singularity 

# cd to the scratch location 
cd /scratch/users/$USER/dnn-unsupervised/bin_main/exp/

########################## DEBUG OUTPUT  ###########################
# grep $SLURM_EXPORT_ENV
echo "log data dir is : ${logdatadir}"
echo "Output data dir is: ${outputdatadir}"
echo "Train data dir is: ${traindatadir}"
echo "Test data dir is: ${testdatadir}"
echo "Patient to analyze is: ${patient}"
echo ${CUDA_VISIBLE_DEVICES}

# print out nvidia's system management interface log
# nvidia-smi

################### SETUP SINGULARITY CONTAINER  ####################
# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER
# singularity pull --name pytorch.simg shub://marcc-hpc/pytorch
singularity pull --name pytorch.simg shub://adam2392/deeplearning_hubs:latest

echo $PWD
echo "Running training model"

# run training here in this
pycmd="python ./slurm/exp001/main.py ${traindatadir} ${testdatadir} --output_data_dir ${outputdatadir} --log_data_dir ${logdatadir} --patient_to_loo ${patient} --expname ${expname}"
# singularity -v exec -B /scratch/ --nv ./pytorch.simg python main.py ${traindatadir} ${testdatadir} ${outputdatadir} ${logdatadir} ${patient} ${expname}
singularity -v -d exec -B /scratch/ --nv ./pytorch.simg python main.py ${traindatadir} ${testdatadir} --output_data_dir ${outputdatadir} --log_data_dir ${logdatadir} --patient_to_loo ${patient} --expname ${expname}

# to run it as a shell
ml cuda/9.0
ml singularity 
singularity -v -d shell --nv pytorch.simg 
traindatadir="/scratch/users/ali39@jhu.edu/data/dnn/traindata_fft/realtng/pipeline/"
testdatadir="/scratch/users/ali39@jhu.edu/data/dnn/traindata_fft/realtng/pipeline/"
patient="id001_bt"
expname="exp001"
logdatadir="/scratch/users/ali39@jhu.edu/data/dnn/logs/$expname/"
outputdatadir="/scratch/users/ali39@jhu.edu/data/dnn/output/$expname/"
python main.py ${traindatadir} ${testdatadir} --output_data_dir ${outputdatadir} --log_data_dir ${logdatadir} --patient_to_loo ${patient} --expname ${expname}
# exit
