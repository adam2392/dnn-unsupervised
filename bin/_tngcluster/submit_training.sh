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
. /soft/miniconda3/activate
source activate dnn


# 1. Prompt user for input that runs the analysis
echo "Begin analysis." # print beginning statement
# NEED TO RUN FOR EZ=0,1,2,3 and varying PZ all once

# For training and modeling with the real data
# tempdatadir='/scratch/users/ali39@jhu.edu/data/dnn/temp/fft_real/id008/'
# outputdatadir='/scratch/users/ali39@jhu.edu/data/dnn/output/fft_real/train_v8/id008/'
# traindatadir='/scratch/users/ali39@jhu.edu/data/dnn/traindata_fft/realtng/'

# For training and modeling with the simulated data
tempdatadir='/home/adamli/data/dnn/temp/fftsim_full/id008/'
outputdatadir='/home/adamli/data/dnn/output/fftsim_full/id008/'
traindatadir='/home/adamli/data/dnn/traindata_fft/expfull/'
testdatadir='/home/adamli/data/dnn/traindata_fft/realtng/'

# For training and modeling with real fragility data 
# tempdatadir='/scratch/users/ali39@jhu.edu/data/dnn/temp/fragilityaux/train_v1/'
# outputdatadir='/scratch/users/ali39@jhu.edu/data/dnn/output/fragilityaux/train_v1/'
# traindatadir='/scratch/users/ali39@jhu.edu/data/output/pert/'
# rawdatadir='/scratch/users/ali39@jhu.edu/data/converted/'

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
# gpu="gpu:$numgpus"
# echo $gpu
nodelist="n29"

# set jobname
jobname="submit_trainpy.log"
## job reqs
walltime=2:00:0

# create export commands
exvars="tempdatadir=${tempdatadir},\
outputdatadir=${outputdatadir},\
traindatadir=${traindatadir},\
rawdatadir=${rawdatadir},\
testdatadir=${testdatadir} "

##
## NEED TO TEST REQUESING THE SPECIFIC GPU NODE ON THE TNG CLUSTER
##

# build basic sbatch command with all params parametrized
sbatcomm="sbatch \
 --time=${walltime} \
 --nodes=${NUM_NODES} \
 --cpus-per-task=${NUM_CPUPERTASK} \
 --job-name=${jobname} \
 --ntasks-per-node=${NUM_PROCSPERNODE} \
 --nodelist=${nodelist} "

# build a scavenger job, gpu job, or other job
printf "Sbatch should run now\n"

echo $sbatcomm $exvars ./run_train_v2.sbatch

${sbatcomm} --export=$exvars ./run_train_v2.sbatch

read -p "Continuing in 0.5 Seconds...." -t 0.5
echo "Continuing ...."
# grep for SLURM_EXPORT_ENV when testing