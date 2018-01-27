#!/bin/bash

ml singularity 

echo "Make sure you cd'ed to the scratch directory!"

cd /scratch/users/$USER/dnn-unsupervised/

# pull tensorflow image onto the marcc-hpc
singularity pull --name tensorflow.simg shub://marcc-hpc/tensorflow 

