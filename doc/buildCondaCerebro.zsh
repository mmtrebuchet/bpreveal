#!/usr/bin/env zsh
#SBATCH --job-name test_rungpu
#SBATCH --ntasks=3
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --time=1:00:00
#SBATCH --output=log%j.log
#SBATCH --partition=gpu
#SBATCH --gres gpu:1

#This is a script that you can run on Cerebro (the cluster at Stowers) 
#to install bpreveal. If you're not on a cluster, you can just run the
#commands in this file, starting with `conda create`. 

#I need to source my .shrc to get conda on the path. 
#You may need to change this to your own shell rc file, or it may
#work without this line for you. 
source /home/cm2363/.zshrc

#Obviously, change the name of the environment to whatever you want.
conda create --yes -n bpreveal-cerebro python=3.10

conda activate bpreveal-cerebro
#Mamba isn't strictly necessary, but it's a lot faster than conda
#when installing lots of packages (looking at you, TensorFlow!). 
#If you don't choose to install mamba, then replace `mamba` in the
#following commands with `conda`. 
conda install -c conda-forge --yes mamba

mamba install --yes -c nvidia cuda-toolkit
mamba install -c conda-forge --yes tensorflow
mamba install -c bioconda --yes pybigwig
mamba install --yes tensorflow-probability
mamba install -c conda-forge --yes matplotlib

#pysam and pybedtools don't have (as of 2023-03-23) Python 3.10 versions
#in the conda repositories. So install them through pip. 
pip install --no-input pysam pybedtools

#Modisco-lite isn't in conda as of 2023-03-23. 
pip install --no-input modisco-lite

#Optional:
#mamba install jupyterlab
#mamba install pycodestyle
#pip install --no-input tensorboard_plugin_profile

#Snakemake doesn't have a python 3.10 version in the conda repositories. 
#pip install --no-input snakemake

#Installed by the above packages:
# h5py
# tqdm
# pandas (by modisco)

