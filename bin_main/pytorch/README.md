# PyTorch

[![https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg](https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg)](https://singularity-hub.org/collections/312)

PyTorch Version: 0.4.0a0

MARCC NVidia GPU and installed drivers at testing time were: K80 & 384.81.

Here is a display of the job submission script: `pytorch_job.sh`:

```
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 1:0:0

module load cuda/9.0
module load singularity

cd mnist

# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER

singularity pull --name pytorch.simg shub://marcc-hpc/pytorch
singularity exec --nv ./pytorch.simg python main.py
```

Clone this repository and submit a job, for example on MARCC Blue Crab:
```
cd /scratch/users/$USER
git clone https://github.com/marcc-hpc/pytorch.git
cd pytorch
sbatch pytorch_job.sh
```

Please open Github issues if you interested in correcting typos, adding
examples, or just providing feedback!

Example taken from https://github.com/pytorch/examples (examples/mnist) LICENSE
is reflected to show attribution; the only modifications are to point to an
empty data folder inside the mnist folder.
