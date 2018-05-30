#!/bin/bash -l

# _out = the log files 
# _logs = the parallel gnu logfile for resuming job at errors 
logdir='_logs'
outdir=_out
# create output directory 
if [ -d "$outdir" ]; then  
	echo "out log directory exists!\n\n"
else
	mkdir $outdir
fi
# create log directory 
if [ -d "$logdir" ]; then  
	echo "log directory for GNU resume runs exists!\n\n"
else
	mkdir $logdir
fi