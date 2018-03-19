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
ml python/3.6.0
ml cuda/9.0
ml singularity 

# activate our conda environment
source activate dnn

# cd to the scratch location 
cd /scratch/users/$USER/dnn-unsupervised/bin/

########################## DEBUG OUTPUT  ###########################
# grep for SLURM_EXPORT_ENV
echo "Temp data dir is : ${tempdatadir}"
echo "Output data dir is: ${outputdatadir}"
echo "Train data dir is: ${traindatadir}"
echo "Raw data dir is: ${rawdatadir}"
echo ${CUDA_VISIBLE_DEVICES}

################### SETUP SINGULARITY CONTAINER  ####################
# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER

echo $PWD
printf "Running training model"
# docker://python:latest
# singularity options: 
# -vvv -d = very verbose / debugging mode
# -B = bind path /scratch /work /data

# for cnn
singularity  exec -B /scratch/ --nv ../tensorflow.simg python ./trainbasecnn/main.py ${outputdatadir} ${tempdatadir} ${traindatadir} ${rawdatadir}

# for training cnn-lstm
# singularity  exec -B /scratch/ --nv ../tensorflow.simg python ./traincnnlstm/main.py ${outputdatadir} ${tempdatadir} ${traindatadir} ${rawdatadir}

# singularity  exec -B /scratch/ --nv ../tensorflow.simg python ./trainfragaux/main.py ${outputdatadir} ${tempdatadir} ${traindatadir} ${rawdatadir}


exit