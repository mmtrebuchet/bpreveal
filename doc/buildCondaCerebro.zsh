#!/usr/bin/env zsh
#
#SBATCH --job-name test_rungpu
#SBATCH --ntasks=3
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --time=1:00:00
#SBATCH --output=log%j.log
#SBATCH --partition=gpu
#SBATCH --gres gpu:1

#I need to source my .shrc to get conda on the path. 
source /home/cm2363/.zshrc
conda create --yes -p /dev/shm/bpreveal-cerebro-tfbase python=3.10

conda activate /dev/shm/bpreveal-cerebro-tfbase
conda install -c conda-forge --yes mamba

mamba install --yes -c nvidia cuda-toolkit
mamba install -c conda-forge --yes tensorflow
mamba install -c bioconda --yes pybigwig
mamba install --yes tensorflow-probability
mamba install -c conda-forge --yes matplotlib

pip install --no-input pysam pybedtools
pip install --no-input modisco-lite

#Optional:
#mamba install jupyterlab
#mamba install pycodestyle
#pip install --no-input tensorboard_plugin_profile
#pip install --no-input snakemake

#Installed by the above packages:
# h5py
# tqdm
# pandas

