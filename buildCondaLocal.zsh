#!/usr/bin/env zsh
#This is a script that you can run on your personal computer
#to install bpreveal.
################
# CHANGE THESE #
################
# The shell that you use.
SHELL=zsh

# I need to source my .shrc to get conda on the path.
# CHANGE this to your own shell rc file, or it may
# work without this line for you.

export SHELL_RC_FILE=/home/cm2363/.zshrc
# The location where you cloned the git repository.
# CHANGE to reflect your directory.
export BPREVEAL_DIR=/n/projects/cm2363/bpreveal

# -p if you're specifying a path, -n if you're specifying a name.
# CHANGE the environment name to your own preference.
export ENV_FLAG=-n
export ENV_NAME=bpreveal-teak

# CHANGE this to conda if you don't have mamba installed.
# (I recommend using mamba; it's way faster.)
export CONDA_BIN=mamba
export PIP_BIN=pip

# Do you want to install Jupyter?
# Options: true or false
export INSTALL_JUPYTER=true

#Do you want the tools used for development?
# These are needed to run the code quality checks and build the html documentation.
# Options: true or false
export INSTALL_DEVTOOLS=true

# Do you want to install pydot and graphviz? This is needed to render an image from showModel.
export INSTALL_PYDOT=true

# Do you want to install some miscellaneous stuff that Charles likes?
# I recommend that you set this to False, as these are not necessary for
# anything in BPReveal.
export INSTALL_MISC=true

######################
# DON'T CHANGE BELOW #
######################

$SHELL ${BPREVEAL_DIR}/etc/buildCondaSuite.zsh
