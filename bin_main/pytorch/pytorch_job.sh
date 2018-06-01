#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:0
#SBATCH --error=%A.err

module load cuda/9.0
module load singularity

# cd mnist
# cd to the scratch location 
cd /scratch/users/$USER/dnn-unsupervised/bin_main/pytorch/mnist/

# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER

singularity pull --name pytorch.simg shub://marcc-hpc/pytorch
singularity exec --nv ./pytorch.simg python main.py #${traindatadir} ${testdatadir} ${outputdatadir}
