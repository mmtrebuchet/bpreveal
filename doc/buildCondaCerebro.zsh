#!/usr/bin/env zsh
#SBATCH --job-name test_rungpu
#SBATCH --ntasks=3
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --time=1:00:00
#SBATCH --output=log%j.log
#SBATCH --partition=gpu
#  #SBATCH --gres gpu:1

#This is a script that you can run on Cerebro (the cluster at Stowers)
#to install bpreveal. If you're not on a cluster, you can just run the
#commands in this file, starting with `conda create`.

#I need to source my .shrc to get conda on the path.
#You may need to change this to your own shell rc file, or it may
#work without this line for you.
source /home/cm2363/.bashrc
source /home/cm2363/.zshrc

#Obviously, change the name of the environment to whatever you want.
conda create --yes -n bpreveal-testing python=3.10

conda activate bpreveal-testing
conda install --yes -c conda-forge mamba
mamba install --yes -c conda-forge cudatoolkit=11.8.0
pip install --no-input nvidia-cudnn-cu11==8.6.0.163
#This garbage is from the Tensorflow install guide. It should put the nvidia libraries on your path, so Tensorflow can find them.
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

#Tensorflow expressly advises against installing with conda.
mamba install --yes -c conda-forge tensorflow-gpu

#Reinstalling cuda-toolkit from the nvidia channel gives you ptxas.
mamba install --yes -c nvidia cuda-toolkit=11.8
mamba install --yes -c conda-forge tensorflow-probability
mamba install --yes -c conda-forge matplotlib
#conda install --yes -c conda-forge chardet
#pysam and pybedtools don't have (as of 2023-03-23) Python 3.10 versions
#in the conda repositories. So install them through pip.
pip install --no-input pysam pybedtools pybigwig


#Optional:
mamba install --yes -c conda-forge jupyterlab
mamba install --yes -c conda-forge pycodestyle
#pip install --no-input tensorboard_plugin_profile

#Modisco-lite isn't in conda as of 2023-03-23.
pip install --no-input modisco-lite

#Snakemake doesn't have a python 3.10 version in the conda repositories.

#pip install --no-input snakemake

#Installed by the above packages:
# h5py
# tqdm
# pandas (by modisco)

