#!/bin/bash
ZSH_RC_FILE="${HOME}/.bashrc" ../buildCondaLocal.sh
eval "$(conda shell.bash hook)"
conda activate bpreveal-teak
BUILDDIR=${READTHEDOCS_OUTPUT:-"_build"} make html