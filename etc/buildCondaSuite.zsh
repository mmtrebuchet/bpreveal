######################
# DON'T CHANGE BELOW #
######################
source ${SHELL_RC_FILE}
conda deactivate  # In case the user forgot.

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
# numba requires an older numpy version, so lock to 2.0.
runAndCheck ${CONDA_BIN} install --yes -c conda-forge "numpy\<2.1"
# Tensorflow expressly advises against installing with conda.
${PIP_BIN} install 'tensorflow[and-cuda]'
check Installing tensorflow

# tensorflow-probability needs tf-keras (at least as of tf 2.17)
runAndCheck ${PIP_BIN} install "tf-keras"
runAndCheck ${PIP_BIN} install 'tensorflow-probability'
checkPackage tensorflow
checkPackage tensorflow_probability


runAndCheck ${CONDA_BIN} install --yes -c conda-forge matplotlib jsonschema cmake h5py \
    tqdm gxx_linux-64 gfortran meson conda-build
checkPackage matplotlib
checkPackage jsonschema
checkPackage h5py
checkPackage tqdm

runAndCheck ${CONDA_BIN} install --yes -c bioconda bedtools pybedtools pybigwig pysam

# pysam, pybigwig, and pybedtools don't have (as of 2024-02-01) Python 3.11
# versions in the conda repositories. So install them through pip.
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
    runAndCheck ${PIP_BIN} install --no-input flake8-bugbear flake8-mutable \
        flake8-print flake8-eradicate flake8-annotations flake8-pep585
fi

if [ "$INSTALL_MISC" = true ] ; then
    # These are just convenience things that I use.
    # They are never required to do anything in BPReveal, but I use them for
    # development and analysis.
    runAndCheck ${CONDA_BIN} install --yes -c conda-forge lazygit git-lfs gprof2dot \
        mpl-scatter-density graphviz
    runAndCheck ${PIP_BIN} install --no-input tabview
fi

if [ "$INSTALL_PYDOT" = true ] ; then
    runAndCheck ${CONDA_BIN} install --yes -c conda-forge graphviz pydot
fi

# 4. Try to build libjaccard.
cd ${BPREVEAL_DIR}/src && make clean && make
check

# 5. Set up bpreveal on your python path.
runAndCheck conda develop ${BPREVEAL_DIR}/pkg


#Add binaries to your path. If you skip this, you can
#always give the full name to the bpreveal tools.
#N.B. If you're manually setting up your environment, running these commands will clobber
#your shell. Once you've run them, exit and restart your shell.
echo "export BPREVEAL_KILL_PATH=${BPREVEAL_DIR}/bin"\
    > ${CONDA_PREFIX}/etc/conda/activate.d/bpreveal_bin_activate.sh
echo "export PATH=\$BPREVEAL_KILL_PATH:\$PATH"\
    >> ${CONDA_PREFIX}/etc/conda/activate.d/bpreveal_bin_activate.sh

    # Add the man pages to the environment. (You still have to build them with cd doc && make man)
echo "export BPREVEAL_MAN_KILL_PATH=${BPREVEAL_DIR}/doc/_build/man"\
    > ${CONDA_PREFIX}/etc/conda/activate.d/bpreveal_man_activate.sh
echo "export MANPATH=\$BPREVEAL_MAN_KILL_PATH:\$MANPATH"\
    >> ${CONDA_PREFIX}/etc/conda/activate.d/bpreveal_man_activate.sh

# Oh no, it's Nvidia garbage. Maybe some day this won't be necessary.
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

# Since Keras 3 has a bunch of breaking changes, force tensorflow
# to use its internal keras version.
# echo "export TF_USE_LEGACY_KERAS=1" \
#     > ${CONDA_PREFIX}/etc/conda/activate.d/legacy_keras_activate.sh

#And add a (very hacky) deactivation command that removes bpreveal from
#your path when you deactivate the environment.
echo "export PATH=\$(echo \$PATH | tr ':' '\n' | grep -v \$BPREVEAL_KILL_PATH | tr '\n' ':')"\
    > ${CONDA_PREFIX}/etc/conda/deactivate.d/bpreveal_bin_deactivate.sh
echo "unset BPREVEAL_KILL_PATH"\
    >> ${CONDA_PREFIX}/etc/conda/deactivate.d/bpreveal_bin_deactivate.sh
# And the same for the man path.
echo "export MANPATH=\$(echo \$MANPATH | tr ':' '\n' | grep -v \$BPREVEAL_MAN_KILL_PATH | tr '\n' ':')"\
    > ${CONDA_PREFIX}/etc/conda/deactivate.d/bpreveal_man_deactivate.sh
echo "unset BPREVEAL_MAN_KILL_PATH"\
    >> ${CONDA_PREFIX}/etc/conda/deactivate.d/bpreveal_man_deactivate.sh
echo "unset XLA_FLAGS" \
    > ${CONDA_PREFIX}/etc/conda/deactivate.d/cuda_xla_deactivate.sh
# Remove bpreveal from LD_LIBRARY_PATH.
echo '\nexport LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed "s|$BPREVEAL_KILL_LD_LIB_PATH||" )'\
    >> ${CONDA_PREFIX}/etc/conda/deactivate.d/cuda_xla_deactivate.sh
# echo "unset TF_USE_LEGACY_KERAS" \
#     > ${CONDA_PREFIX}/etc/conda/deactivate.d/legacy_keras_deactivate.sh


echo "*-----------------------------------*"
echo "| BPReveal installation successful. |"
echo "|           (probably...)           |"
echo "*-----------------------------------*"

