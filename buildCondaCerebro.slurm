#!/usr/bin/env zsh
#SBATCH --job-name buildBpreveal
#SBATCH --ntasks=3
#SBATCH --nodes=1
#SBATCH --mem=10gb
#SBATCH --time=1:00:00
#SBATCH --output=log%j.log
#SBATCH --partition=gpu
#SBATCH --gres gpu:0

# This is a script that you can run on Cerebro (the cluster at Stowers)
# to install bpreveal. If you're not on a cluster, see the buildCondaLocal.zsh
# script.

################
# CHANGE THESE #
################

# I need to source my .shrc to get conda on the path.
# CHANGE this to your own shell rc file, or it may
# work without this line for you.

source /home/cm2363/.zshrc
# The location where you cloned the git repository.
# CHANGE to reflect your directory.
BPREVEAL_DIR=/n/projects/cm2363/bpreveal

# -p if you're specifying a path, -n if you're specifying a name.
# CHANGE the environment name to your own preference.
ENV_FLAG=-n
ENV_NAME=bpreveal-cerebro

# CHANGE this to conda if you don't have mamba installed.
# (I recommend using mamba; it's way faster.)
CONDA_BIN=mamba
PIP_BIN=pip

# Do you want to install Jupyter?
# Options: true or false
INSTALL_JUPYTER=true

#Do you want the tools used for development?
# These are needed to run the code quality checks and build the html documentation.
# Options: true or false
INSTALL_DEVTOOLS=true

# Do you want to install pydot and graphviz? This is needed to render an image from showModel.
INSTALL_PYDOT=true

######################
# DON'T CHANGE BELOW #
######################

conda deactivate # In case the user forgot.

PYTHON_VERSION=3.12

runAndCheck() {
    currentCommand=$@
    echo "EXECUTING COMMAND: [[$currentCommand]]"
    eval "$currentCommand"
    errorVal=$?
    if [ $errorVal -eq 0 ]; then
        echo "SUCCESSFULLY EXECUTED: [[$currentCommand]]"
    else
        echo "ERROR DETECTED: Command [[$currentCommand]] on line $BASH_LINENO exited with status $errorVal"
        exit 1
    fi
}

check() {
    errorVal=$?
    if [ $errorVal -eq 0 ]; then
        echo "Command completed successfully"
    else
        echo "ERROR DETECTED: Command $@ exited with status $errorVal"
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


runAndCheck ${CONDA_BIN} create --yes ${ENV_FLAG} ${ENV_NAME} python=${PYTHON_VERSION}

runAndCheck conda activate ${ENV_NAME}

#Make sure we have activated an environment with the right python.
runAndCheck python3 --version \| grep -q "${PYTHON_VERSION}"
# Tensorflow expressly advises against installing with conda.

${PIP_BIN} install 'tensorflow[and-cuda]'
check install tensorflow

runAndCheck ${PIP_BIN} install 'tensorflow-probability'
checkPackage tensorflow
checkPackage tensorflow_probability

runAndCheck ${PIP_BIN} install 'tf-keras~=2.16'

runAndCheck ${CONDA_BIN} install --yes -c conda-forge matplotlib
checkPackage matplotlib

#jsonschema is used to validate input jsons.
runAndCheck ${CONDA_BIN} install --yes -c conda-forge jsonschema
checkPackage jsonschema

# cmake is necessary to build the wheels for modiscolite.
runAndCheck ${CONDA_BIN} install --yes -c conda-forge cmake

# These will have been installed by previous code, but doesn't hurt to explicitly list them.
runAndCheck ${CONDA_BIN} install --yes -c conda-forge h5py
checkPackage h5py

runAndCheck ${CONDA_BIN} install --yes -c conda-forge tqdm
checkPackage tqdm

# Before building stuff with pip, we need to make sure we have a compiler installed.
runAndCheck ${CONDA_BIN} install --yes -c conda-forge gxx_linux-64

runAndCheck ${CONDA_BIN} install --yes -c bioconda bedtools

# pysam, pybigwig, and pybedtools don't have (as of 2024-02-01) Python 3.11
# versions in the conda repositories. So install them through pip.
runAndCheck ${PIP_BIN} install --no-input pysam pybedtools pybigwig
checkPackage pybedtools
checkPackage pyBigWig
checkPackage pysam

# Modisco-lite isn't in conda as of 2024-02-01.
runAndCheck ${PIP_BIN} install --no-input modisco-lite

# 1. Install jupyter lab.
if [ "$INSTALL_JUPYTER" = true ] ; then
    runAndCheck ${CONDA_BIN} install --yes -c conda-forge jupyterlab pandoc
fi


# 2. Install things for development (used in development to check code style,
# never needed to run bpreveal.)
if [ "$INSTALL_DEVTOOLS" = true ] ; then
    runAndCheck ${CONDA_BIN} install --yes -c conda-forge flake8 pydocstyle \
        pylint sphinx sphinx_rtd_theme sphinx-argparse sphinx-autodoc-typehints coverage
fi

if [ "$INSTALL_PYDOT" = true ] ; then
    runAndCheck ${CONDA_BIN} install --yes -c conda-forge graphviz pydot
fi

# 4. Try to build libjaccard.
runAndCheck ${CONDA_BIN} install --yes -c conda-forge gfortran meson

cd ${BPREVEAL_DIR}/src && make clean && make
check

# 5. Set up bpreveal on your python path.
runAndCheck ${CONDA_BIN} install --yes -c conda-forge conda-build

runAndCheck conda develop ${BPREVEAL_DIR}/pkg


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

cat >>  ${CONDA_PREFIX}/etc/conda/activate.d/cuda_xla_activate.sh << EOF


NVIDIA_DIR=\$(dirname \$(dirname \$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)")))

pathAccum=""

for dir in \$NVIDIA_DIR/*; do
    if [ -d "\$dir/lib" ]; then
        pathAccum="\$dir/lib:\$pathAccum"
    fi
done

export BPREVEAL_KILL_LD_LIB_PATH=\$pathAccum
export LD_LIBRARY_PATH="\${pathAccum}\${LD_LIBRARY_PATH}"

EOF


echo "export TF_USE_LEGACY_KERAS=1" \
    > ${CONDA_PREFIX}/etc/conda/activate.d/legacy_keras_activate.sh

#And add a (very hacky) deactivation command that removes bpreveal from
#your path when you deactivate the environment.
echo "export PATH=\$(echo \$PATH | tr ':' '\n' | grep -v \$BPREVEAL_KILL_PATH | tr '\n' ':')"\
    > ${CONDA_PREFIX}/etc/conda/deactivate.d/bpreveal_bin_deactivate.sh
echo "unset BPREVEAL_KILL_PATH"\
    >> ${CONDA_PREFIX}/etc/conda/deactivate.d/bpreveal_bin_deactivate.sh
echo "unset XLA_FLAGS" \
    > ${CONDA_PREFIX}/etc/conda/deactivate.d/cuda_xla_deactivate.sh
echo '\nexport LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed "s|$BPREVEAL_KILL_LD_LIB_PATH||" )'\
    >> ${CONDA_PREFIX}/etc/conda/deactivate.d/cuda_xla_deactivate.sh
echo "unset TF_USE_LEGACY_KERAS" \
    > ${CONDA_PREFIX}/etc/conda/deactivate.d/legacy_keras_deactivate.sh


echo "*-----------------------------------*"
echo "| BPReveal installation successful. |"
echo "|           (probably...)           |"
echo "*-----------------------------------*"

