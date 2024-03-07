#!/usr/bin/env zsh

# This installs a minimal environment, just enough to build the documentation.

source "${HOME}/.zshrc"

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

PYTHON_VERSION=3.11

check() {
    errorVal=$?
    if [ $errorVal -eq 0 ]; then
        echo "Command completed successfully"
    else
        echo "ERROR DETECTED: Last command exited with status $errorVal"
        exit
    fi
}


${CONDA_BIN} create --yes ${ENV_FLAG} ${ENV_NAME} python=${PYTHON_VERSION}
check

conda activate ${ENV_NAME}
check

#Make sure we have activated an environment with the right python.
python3 --version | grep -q "${PYTHON_VERSION}"
check

${CONDA_BIN} install --yes -c conda-forge \
    jsonschema sphinx sphinx_rtd_theme sphinx-autodoc-typehints \
    sphinx-argparse conda-build numpy matplotlib h5py scipy
check

pip install pybedtools pysam pyBigWig
check

cd ${BPREVEAL_DIR}/src && make schemas
check

conda develop ${BPREVEAL_DIR}/pkg
check
