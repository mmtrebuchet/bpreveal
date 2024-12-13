#!/usr/bin/env bash

# This installs a minimal environment, just enough to build the documentation.
# It then builds the documentation

ENV_FLAG=-n
ENV_NAME=bpreveal-docbuild

CONDA_BIN=conda

######################
# DON'T CHANGE BELOW #
######################

SCRIPT_NAME=$(readlink -f -- "$0" )
DOC_DIR_NAME=$(dirname -- "${SCRIPT_NAME}" )
BPREVEAL_DIR=$(dirname -- "${DOC_DIR_NAME}" )
echo "Installing from directory ${BPREVEAL_DIR}"

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

eval "$(conda shell.bash hook)"

runAndCheck ${CONDA_BIN} create --yes ${ENV_FLAG} ${ENV_NAME} python=${PYTHON_VERSION}

runAndCheck conda activate ${ENV_NAME}

#Make sure we have activated an environment with the right python.
runAndCheck python3 --version \| grep -q "${PYTHON_VERSION}"

runAndCheck ${CONDA_BIN} install --yes -c conda-forge \
    jsonschema sphinx sphinx_rtd_theme sphinx-autodoc-typehints \
    sphinx-argparse conda-build numpy matplotlib h5py scipy \
    gfortran gxx_linux-64 meson

runAndCheck ${CONDA_BIN} install --yes -c bioconda pybedtools

runAndCheck pip install pysam pyBigWig

runAndCheck cd ${BPREVEAL_DIR}/src \&\& make schemas

runAndCheck cd ${BPREVEAL_DIR}/src \&\& make

runAndCheck conda develop ${BPREVEAL_DIR}/pkg

runAndCheck cd ${BPREVEAL_DIR}/doc \&\& make html