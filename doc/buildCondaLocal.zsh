#!/usr/bin/env zsh
#This is a script that you can run on your personal computer
#to install bpreveal.
################
# CHANGE THESE #
################

#The location where you cloned the git repository.
#CHANGE to reflect your directory.
BPREVEAL_DIR=/n/projects/cm2363/bpreveal

# -p if you're specifying a path, -n if you're specifying a name.
# CHANGE the environment name to your own preference.
ENV_FLAG=-n
ENV_NAME=bpreveal-teak
#If you want an enironment that you load like a normal conda environment,
#change this to:
#ENV_FLAG=-n
#ENV_NAME=bpreveal-testing

#CHANGE this to conda if you don't have mamba installed.
CONDA_BIN=mamba

#I need to source my .shrc to get conda on the path.
#CHANGE this to your own shell rc file, or it may
#work without this line for you.
source /home/cm2363/.zshrc


######################
# DON'T CHANGE BELOW #
######################


check() {
    errorVal=$?
    if [ $errorVal -eq 0 ]; then
        echo "Command completed successfully."
    else
        echo "ERROR DETECTED: Last command exited with status $errorVal"
        exit
    fi
}


#If you want to do CWM scanning, you may need to build the libjaccard
#library. If you get errors regarding jaccard importing,
#go into the src/ directory, remove jaccard.cpython-310-x86_64-linux-gnu.so,
#and then run make with no arguments. (The conda environment will need to
#be loaded to run this makefile.)


#Obviously, change the name of the environment to whatever you want.
${CONDA_BIN} create --yes ${ENV_FLAG} ${ENV_NAME} python=3.11
check
conda activate ${ENV_NAME}
check

#Tensorflow expressly advises against installing with conda.
#I'm gonna do it anyway, because it works for me.
${CONDA_BIN} install --yes -c conda-forge tensorflow-gpu
check
#You have to install the nvidia-toolkit manually to enable optimizations
#for tensorflow.
${CONDA_BIN} install --yes -c "nvidia/label/cuda-11.8.0" cuda-toolkit
check

${CONDA_BIN} install --yes -c conda-forge tensorflow-probability
check
${CONDA_BIN} install --yes -c conda-forge matplotlib
check
${CONDA_BIN} install --yes -c conda-forge cmake
check

#pysam and pybedtools don't have (as of 2023-03-23) Python 3.10 versions
#in the conda repositories. So install them through pip.
pip install --no-input pysam pybedtools pybigwig
check

#Modisco-lite isn't in conda as of 2023-03-23.
conda install --yes -c conda-forge cmake
check
pip install --no-input modisco-lite
check

#Optional:
${CONDA_BIN} install --yes -c conda-forge jupyterlab
check
${CONDA_BIN} install --yes -c conda-forge pycodestyle
check
#pip install --no-input tensorboard_plugin_profile

#Snakemake doesn't have a python 3.10 version in the conda repositories.
pip install --no-input snakemake
check

#Installed by the above packages:
# h5py
# tqdm
# pandas (by modisco)


#Optional:
#1. Try to build libjaccard.
cd ${BPREVEAL_DIR}/src && make clean && make
check

#2. Set up bpreveal on your python path.
${CONDA_BIN} install -c conda-forge --yes conda-build
check
conda develop ${BPREVEAL_DIR}/pkg
check

#Add binaries to your path. If you skip this, you can
#always give the full name to the bpreveal tools.

#echo "export BPREVEAL_OLD_PATH=\$PATH" > ${CONDA_PREFIX}/etc/conda/activate.d/bpreveal_bin_activate.sh
echo "export BPREVEAL_KILL_PATH=${BPREVEAL_DIR}/bin" > ${CONDA_PREFIX}/etc/conda/activate.d/bpreveal_bin_activate.sh
echo "export PATH=\$BPREVEAL_KILL_PATH:\$PATH" >> ${CONDA_PREFIX}/etc/conda/activate.d/bpreveal_bin_activate.sh

echo "export PATH=\$(echo \$PATH | tr ':' '\n' | grep -v \$BPREVEAL_KILL_PATH | tr '\n' ':')"\
    > ${CONDA_PREFIX}/etc/conda/deactivate.d/bpreveal_bin_deactivate.sh
echo "unset BPREVEAL_KILL_PATH"\
    >> ${CONDA_PREFIX}/etc/conda/deactivate.d/bpreveal_bin_deactivate.sh

