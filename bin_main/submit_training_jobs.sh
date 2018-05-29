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
module unload git
ml python/3.6.5
ml anaconda-python/3.6
source activate dnn
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

#### Create all logging directories if needed
outdir=_out
# create output directory 
if [ -d "$outdir" ]; then  
	echo "Out log directory exists!\n\n"
else
	mkdir $outdir
fi

########################### 2. Define Slurm Parameters ###########################
NUM_PROCSPERNODE=6  	# number of processors per node (1-24). Use 24 for GNU jobs.
NUM_NODES=1				# number of nodes to request
NUM_CPUPERTASK=1

# set the parameters for the GPU partition
partition=gpu 	# debug, shared, unlimited, parallel, gpu, lrgmem, scavenger
numgpus=1
gpu="gpu:$numgpus"
echo $gpu

# set jobname
jobname="train_${patient}_${expname}.log"
## job reqs
walltime=5:00:0

# partition=gpu
walltime=0:30:0

for patient in $patients; do
	# create export commands
	exvars="--export=logdatadir=${logdatadir},\
outputdatadir=${outputdatadir},\
traindatadir=${traindatadir},\
testdatadir=${testdatadir},\
patient=${patient},\
expname=${expname} "

	# build basic sbatch command with all params parametrized
	sbatcomm="sbatch \
	 --time=${walltime} \
	 --nodes=${NUM_NODES} \
	 --cpus-per-task=${NUM_CPUPERTASK} \
	 --job-name=${jobname} \
	 --ntasks-per-node=${NUM_PROCSPERNODE} \
	 --partition=${partition} \
	 --gres=${gpu} "

	# build a scavenger job, gpu job, or other job
	echo "Sbatch should run now"
	echo $sbatcomm $exvars ./slurm/run_train_pytorch.sbatch

	${sbatcomm} $exvars ./slurm/run_train_pytorch.sbatch

	read -p "Continuing in 0.5 Seconds...." -t 0.5
	echo "Continuing ...."

done