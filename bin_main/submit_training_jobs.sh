#!/bin/bash -l

##############################################################
#
# Shell script for submitting parallel python jobs on SLURM 
# cluster with nodes, CPUS, tasks, GPUs
#
##############################################################

################################### 1. READ USER INPUT ###########################################
patients=(
'id001_bt')
# id002_sd
# id003_mg 
# id004_bj id005_ft
# id006_mr id007_rd id008_dmc
# id009_ba id010_cmn 
# id011_gr id013_lk id014_vc id015_gjl
# id016_lm id017_mk id018_lo id020_lma')

## load in the modules for this run -> python, matlab, etc.
# module unload git
ml python/3.6.5
# ml anaconda-python/3.6
# source activate dnn
module list

# Pause before running to check
# printf "About to run on patients (press enter to continue): $patients" 
# read answer

expname="exp001"
## For training and modeling with the simulated data
# traindatadir="/scratch/users/ali39@jhu.edu/data/dnn/traindata_fft/exp001/"
traindatadir="/scratch/users/ali39@jhu.edu/data/dnn/traindata_fft/realtng/pipeline/"
testdatadir="/scratch/users/ali39@jhu.edu/data/dnn/traindata_fft/realtng/pipeline/"
# logs for the training logs, and outputdata directory for final summary
logdatadir="/scratch/users/ali39@jhu.edu/data/dnn/logs/$expname/" 			
outputdatadir="/scratch/users/ali39@jhu.edu/data/dnn/output/$expname/"

echo "These are the data directories: "
echo "Temp datadir: $logdatadir "
echo "Output datadir: $outputdatadir "
echo "Results datadir: $datadir "
echo "Testing datadir is: $testdatadir"

# run setup of a slurm job
setup="./config/slurm/setup.sh"
. $setup

########################### 2. Define Slurm Parameters ###########################
gpu_debug_config="./config/slurm/gpu_debug_jobs.txt"
gpu_config="./config/slurm/gpu_jobs.txt"

for patient in $patients; do
	echo $patient 

	# create export commands
	exvars="--export=logdatadir=${logdatadir},\
outputdatadir=${outputdatadir},\
traindatadir=${traindatadir},\
testdatadir=${testdatadir},\
patient=${patient},\
expname=${expname} "

	# build a scavenger job, gpu job, or other job
	# set jobname
	jobname="train_${patient}_${expname}.log"
	sbatchcomm=$(cat $gpu_debug_config)
	sbatchcomm="$sbatchcomm --job-name=${jobname}"

	# build a scavenger job, gpu job, or other job
	echo "Sbatch should run now"
	echo $sbatchcomm $exvars ./run_train_pytorch.sbatch

	cd /scratch/users/$USER/dnn-unsupervised/bin_main/exp/
	${sbatchcomm} $exvars ./run_train_pytorch.sbatch

	read -p "Continuing in 0.5 Seconds...." -t 0.5
	echo "Continuing ...."
done
