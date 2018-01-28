Bootstrap: docker
From: tensorflow/tensorflow:latest-gpu

%environment
  # use bash as default shell
  SHELL=/bin/bash
  export SHELL

%setup
  # runs on host - the path to the image is $SINGULARITY_ROOTFS

%post
  # post-setup script

  # load environment variables
  . /environment

  # use bash as default shell
  echo 'SHELL=/bin/bash' >> /environment

  # make environment file executable
  chmod +x /environment

  # default mount paths
  mkdir /scratch /data 

  # load in extra packages for python
  apt-get update && apt-get -y install locales
  locale-gen en_US.UTF-8
  apt-get install -y git wget python3-dev python3-pip
  apt-get clean

  apt-get install -y libcupti-dev
  pip3 install --upgrade pip
  pip3 install keras
  pip3 install numpy scipy scikit-learn pandas

%runscript
  # executes with the singularity run command
  # delete this section to use existing docker ENTRYPOINT command

%test
  # test that script is a success