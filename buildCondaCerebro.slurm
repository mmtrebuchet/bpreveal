#!/usr/bin/env zsh
#SBATCH --job-name buildBpreveal
#SBATCH --ntasks=3
#SBATCH --nodes=1
#SBATCH --mem=5gb
#SBATCH --time=1:00:00
#SBATCH --output=log%j.log
#SBATCH --partition=gpu
#SBATCH --gres gpu:a100_3g.20gb:0

# This is a script that you can run on Cerebro (the cluster at Stowers)
# to install bpreveal. If you're not on a cluster, see the buildCondaLocal.zsh
# script.

################
# CHANGE THESE #
################

# I need to source my .shrc to get conda on the path.
# CHANGE this to your own shell rc file, or it may
# work without this line for you.
source /home/cm2363/.bashrc
source /home/cm2363/.zshrc
# The location where you cloned the git repository.
# CHANGE to reflect your directory.
BPREVEAL_DIR=/n/projects/cm2363/bpreveal

# -p if you're specifying a path, -n if you're specifying a name.
# CHANGE the environment name to your own preference.
ENV_FLAG=-p
ENV_NAME=/n/projects/cm2363/public-bpreveal/4.0.0/env

# CHANGE this to conda if you don't have mamba installed.
CONDA_BIN=mamba
PIP_BIN=pip

# Do you want to install Jupyter?
# Options: true or false
INSTALL_JUPYTER=true

#Do you want the tools used for development?
# These are needed to run the code quality checks and build the html documentation.
# Options: true or false
INSTALL_DEVTOOLS=true

# Do you want to install Snakemake?
INSTALL_SNAKEMAKE=true


######################
# DON'T CHANGE BELOW #
######################

check() {
    errorVal=$?
    if [ $errorVal -eq 0 ]; then
        echo "Command completed successfully"
    else
        echo "ERROR DETECTED: Last command exited with status $errorVal"
        exit
    fi
}

checkPackage() {
    python3 -c "import $1"
    errorVal=$?
    if [ $errorVal -eq 0 ]; then
        echo "$1 imported successfully."
    else
        echo "ERROR DETECTED: Failed to import $1"
        exit
    fi
}


${CONDA_BIN} create --yes ${ENV_FLAG} ${ENV_NAME} python=3.11
check

conda activate ${ENV_NAME}
check
# Tensorflow expressly advises against installing with conda.

${PIP_BIN} install 'tensorflow[and-cuda]'
check
${PIP_BIN} install tensorflow-probability
check
checkPackage tensorflow
checkPackage tensorflow_probability

${CONDA_BIN} install --yes -c conda-forge matplotlib
check
checkPackage matplotlib

#jsonschema is used to validate input jsons.
${CONDA_BIN} install --yes -c conda-forge jsonschema
check
checkPackage jsonschema

# cmake is necessary to build the wheels for modiscolite.
${CONDA_BIN} install --yes -c conda-forge cmake
check

# These will have been installed by previous code, but doesn't hurt to explicitly list them.
${CONDA_BIN} install --yes -c conda-forge h5py
check
checkPackage h5py

${CONDA_BIN} install --yes -c conda-forge tqdm
check
checkPackage tqdm

# Before building stuff with pip, we need to make sure we have a compiler installed.
${CONDA_BIN} install --yes -c conda-forge gxx_linux-64
check

${CONDA_BIN} install --yes -c bioconda bedtools
check

# pysam, pybigwig, and pybedtools don't have (as of 2023-03-23) Python 3.11
# versions in the conda repositories. So install them through pip.
${PIP_BIN} install --no-input pysam pybedtools pybigwig
check
checkPackage pybedtools
checkPackage pyBigWig
checkPackage pysam

# Modisco-lite isn't in conda as of 2023-03-23.
${PIP_BIN} install --no-input modisco-lite
check

# 1. Install jupyter lab.
if [ "$INSTALL_JUPYTER" = true ] ; then
    ${CONDA_BIN} install --yes -c conda-forge jupyterlab
    check
    ${CONDA_BIN} install --yes -c conda-forge pandoc
    check
fi


# 2. Install flake8 (used in development to check code style,
# never needed to run bpreveal.)
if [ "$INSTALL_DEVTOOLS" = true ] ; then
    ${CONDA_BIN} install --yes -c conda-forge flake8
    check
    ${CONDA_BIN} install --yes -c conda-forge pydocstyle
    check
    ${CONDA_BIN} install --yes -c conda-forge pylint
    check
    ${CONDA_BIN} install --yes -c conda-forge sphinx
    check
    ${CONDA_BIN} install --yes -c conda-forge sphinx_rtd_theme
    check
    ${CONDA_BIN} install --yes -c conda-forge sphinx-argparse
    check
    ${CONDA_BIN} install --yes -c conda-forge sphinx-autodoc-typehints
    check
fi

if [ "$INSTALL_SNAKEMAKE" = true ] ; then
    # 3. Snakemake doesn't have a python 3.10 version in the conda repositories, so install
    # it with pip.
    ${PIP_BIN} install --no-input snakemake
    check
fi


# 4. Try to build libjaccard.
${CONDA_BIN} install --yes -c conda-forge gfortran
check
cd ${BPREVEAL_DIR}/src && make clean && make
check

# 5. Set up bpreveal on your python path.
${CONDA_BIN} install --yes -c conda-forge conda-build
check

conda develop ${BPREVEAL_DIR}/pkg
check


#Add binaries to your path. If you skip this, you can
#always give the full name to the bpreveal tools.
#N.B. If you're manually setting up your environment, running these commands will clobber
#your shell. Once you've run them, exit and restart your shell.
echo "export BPREVEAL_KILL_PATH=${BPREVEAL_DIR}/bin"\
    > ${CONDA_PREFIX}/etc/conda/activate.d/bpreveal_bin_activate.sh
echo "export PATH=\$BPREVEAL_KILL_PATH:\$PATH"\
    >> ${CONDA_PREFIX}/etc/conda/activate.d/bpreveal_bin_activate.sh

echo "export XLA_FLAGS=\"--xla_gpu_cuda_data_dir=${CONDA_PREFIX}\"" \
    > ${CONDA_PREFIX}/etc/conda/activate.d/cuda_xla_activate.sh

#And add a (very hacky) deactivation command that removes bpreveal from
#your path when you deactivate the environment.
echo "export PATH=\$(echo \$PATH | tr ':' '\n' | grep -v \$BPREVEAL_KILL_PATH | tr '\n' ':')"\
    > ${CONDA_PREFIX}/etc/conda/deactivate.d/bpreveal_bin_deactivate.sh
echo "unset BPREVEAL_KILL_PATH"\
    >> ${CONDA_PREFIX}/etc/conda/deactivate.d/bpreveal_bin_deactivate.sh
echo "unset XLA_FLAGS" \
    > ${CONDA_PREFIX}/etc/conda/deactivate.d/cuda_xla_deactivate.sh


echo "*-----------------------------------*"
echo "| BPReveal installation successful. |"
echo "|           (probably...)           |"
echo "*-----------------------------------*"

