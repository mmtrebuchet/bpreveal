
Installation
============

Requirements
------------

This program requires just a few libraries. I provide a buildCondaLocal.zsh
script that you can run on your own machine to automatically build an
environment, or you can install the dependencies yourself with conda or pip.

python >= 3.12
    This project uses features that were introduced in Python 3.11 heavily, so
    earlier Python versions will not work.
tfmodisco
    Used to identify motifs. (You may need to install cmake to build the wheel for
    modiscolite.)
pyBigWig
    Used to read in data files.
pysam
    used to read in fasta files.
pyBedTools
    Used to read in region files. You will also need to make sure you have
    BedTools installed to use pyBedTools.
tensorflow
    Does the heavy work of machine learning. Tensorflow 2.16 added a bunch of crummy
    backwards-incompatible behavior and some of the internal API had to be restructured,
    this was done in BPReveal 4.2.0.
tensorflow-probability
    Used to create the multinomial loss function.
matplotlib
    Used to make pretty plots.
tqdm
    Used to create progress bars.
h5py
    Used to read and write data files.
jsonschema
    Used to validate the json files given to the Main CLI programs.
gcc, gfortran, meson
    You'll need a C compiler and a Fortran compiler to build the
    Jaccard and ushuffle libraries, and as of Python 3.12 you also
    need the meson build system.

Optional
^^^^^^^^

While not strictly necessary, the following packages are very useful:

jupyterlab
    For interactive data wrangling.
pandoc
    Used to export Jupyter notebooks as pdfs.
pydot, graphviz
    These are only necessary to use showModel.py, which is deprecated and
    will be removed in BPReveal 6.0.0.


Development
^^^^^^^^^^^

Additionally, for development, there are a few libraries that I find useful:

flake8, pydocstyle, pylint
    Used to check the code for style issues.
flake8-bugbear, flake8-mutable, flake8-print, flake8-eradicate, flake8-annotations, flake8-pep585
    Obsessive style checking.
    BPReveal gets a perfect score from these very aggressive linters.
sphinx, sphinx_rtd_theme, sphinx-argparse, sphinx-autodoc-typehints
    Used to generate the documentation you're reading right now.
coverage
    Used to check code coverage.

Setup
-----

Once you have the requisite packages, you need to do a bit more setup to get
BPReveal fully installed. If you use the buildCondaLocal.zsh script, you can
skip this because the script takes care of this part.

1. In the ``src/`` directory of BPReveal, run ``make clean && make`` to build
   the ``libjaccard``, ``libslide``, and ``libushuffle`` libraries.
   This step also generates the json schemas
   that are used to check your inputs for bugs.
2. Add the ``bin/`` directory to your path. This will be something like::

    export PATH=/path/to/bpreveal/bin:$PATH

3. Add the ``pkg/`` directory to your python path. The easy way to do this is with
   ``conda develop``::

    conda develop /path/to/bpreveal/pkg

4. (Optional) Build the documentation in the ``doc/`` directory::

    cd /path/to/bpreveal/doc && make html latexpdf

..
    Copyright 2022-2025 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.
