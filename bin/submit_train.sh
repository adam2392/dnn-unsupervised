#!/bin/bash -l

##############################################################
#
# Shell script for submitting parallel python jobs on SLURM 
# to run jobs for training deep neural networks.
# 
# - trainbasecnn/ for training a baseline CNN on real data and 
# simulated data
#
##############################################################
# ml python/3.6.0
# ml cuda/9.0
module unload git
ml anaconda-python/3.6
source activate dnn


# 1. Prompt user for input that runs the analysis
echo "Begin analysis." # print beginning statement
# NEED TO RUN FOR EZ=0,1,2,3 and varying PZ all once

# Pause before running to check
tempdatadir='/scratch/users/ali39@jhu.edu/data/dnn/temp/fft_real/'
outputdatadir='/scratch/users/ali39@jhu.edu/data/dnn/output/fft_real/train_v8/'
traindatadir='/scratch/users/ali39@jhu.edu/data/dnn/traindata_fft/'

# /scratch/users/ali39@jhu.edu
printf "\nThis is the data directories: \n"
printf "Temp datadir: $tempdatadir \n"
printf "Output datadir: $outputdatadir \n"
printf "Train datadir: $traindatadir \n"
printf "\n"

#### Create all logging directories if needed
# _gnuerr = all error logs for sbatch gnu runs %A.out 
# _gnuout = all output logs for sbatch gnu runs %A.out 
# _logs = the parallel gnu logfile for resuming job at errors 
outdir=_out
# create output directory 
if [ -d "$outdir" ]; then  
	echo "Out log directory exists!\n\n"
else
	mkdir $outdir
fi

# 2. Define Slurm Parameters
NUM_PROCSPERNODE=6  	# number of processors per node (1-24). Use 24 for GNU jobs.
NUM_NODES=1				# number of nodes to request
NUM_CPUPERTASK=1

# set the parameters for the GPU partition
partition=gpu 	# debug, shared, unlimited, parallel, gpu, lrgmem, scavenger
numgpus=1
gpu="gpu:$numgpus"
echo $gpu

# set jobname
jobname="submit_trainpy.log"
## job reqs
walltime=5:45:0

# create export commands
exvars="tempdatadir=${tempdatadir},\
outputdatadir=${outputdatadir},\
traindatadir=${traindatadir} "

# build basic sbatch command with all params parametrized
sbatcomm="sbatch \
 --time=${walltime} \
 --nodes=${NUM_NODES} \
 --cpus-per-task=${NUM_CPUPERTASK} \
 --job-name=${jobname} \
 --ntasks-per-node=${NUM_PROCSPERNODE} \
 --partition=${partition}
 --gres=${gpu} "

# build a scavenger job, gpu job, or other job
printf "Sbatch should run now\n"

echo $sbatcomm $exvars ./run_train.sbatch

${sbatcomm} --export=$exvars ./run_train.sbatch

read -p "Continuing in 0.5 Seconds...." -t 0.5
echo "Continuing ...."
# grep for SLURM_EXPORT_ENV when testing