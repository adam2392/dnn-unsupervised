#!/bin/bash
#SBATCH
#SBATCH --mem=50G
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
cd /scratch/users/$USER/dnn-unsupervised/binmaster/

########################## DEBUG OUTPUT  ###########################
# grep $SLURM_EXPORT_ENV
echo "log data dir is : ${logdatadir}"
echo "Output data dir is: ${outputdatadir}"
echo "Train data dir is: ${traindatadir}"
echo "Test data dir is: ${testdatadir}"
echo "Model weights/model dir is: ${model_dir}"
echo "Patient to analyze is: ${patient}"
echo "Iteration is: ${iteration}"
echo "Number of CUDA available devices: ${CUDA_VISIBLE_DEVICES}"
# print out nvidia's system management interface log
# nvidia-smi

################### SETUP SINGULARITY CONTAINER  ####################
# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER
singularity pull --name tensorflow.simg shub://adam2392/deeplearning_hubs:keras

echo $PWD
echo "Running training model"

# run training here in this
# python ./hpc_util/baseline_exps/exp_base.py \
# singularity exec -B /scratch/ --nv ./tensorflow.simg \
# python ./hpc_util/eznet/exp_baseline.py \
# 	${traindatadir} ${testdatadir} \
# 	--output_data_dir ${outputdatadir} \
# 	--log_data_dir ${logdatadir} \
# 	--model_dir ${model_dir} \
# 	--patient_to_loo ${patient} \
# 	--expname ${expname}

singularity exec -B /scratch/ --nv ./tensorflow.simg \
python ./hpc_util/baseline_exps/exp_base.py \
${traindatadir} ${testdatadir} \
	--output_data_dir ${outputdatadir} \
	--log_data_dir ${logdatadir} \
	--model_dir ${model_dir} \
	--patient_to_loo ${patient} \
	--expname ${expname}

exit
