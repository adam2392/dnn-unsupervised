#!/bin/bash

#SBATCH
#SBATCH -p gpu # or -p parallel
#SBATCH --mail-type=end
#SBATCH --mail-user=ali39@jhu.edu
#SBATCH --output=_out/%A.out 
#SBATCH --error=_out/%A.err

#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

# Author: Adam Li (ali39@jhu.edu).
# Created on 2017-10-31. 
#---------------------------------------------------------------------
# SLURM job script to run serial Python
# on TNG Cluster
#---------------------------------------------------------------------
module unload git
ml anaconda-python/3.6
source activate dnn

ml python/3.6.0
ml cuda/9.0
ml singularity/2.4

# grep for SLURM_EXPORT_ENV
echo ${tempdatadir}
echo ${outputdatadir}
echo ${CUDA_VISIBLE_DEVICES}

# pull tensorflow image onto the marcc-hpc
singularity pull --name tensorflow shub://marcc-hpc/tensorflow

# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER

# create log directory 
logdir='_logs'
if [ -d "$logdir" ]; then  
	echo "log directory exists!"
else
	mkdir $logdir
fi
# set jobname
logfilename="submit_trainpy.log"

printf "Running training model"
singularity exec --nv ./tensorflow python ./test.py ${outputdatadir} ${tempdatadir}
# python ./main.py ${outputdatadir} ${tempdatadir} ${CUDA_VISIBLE_DEVICES}

exit