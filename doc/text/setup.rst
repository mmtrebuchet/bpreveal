
Installation
============

This program requires just a few libraries. I provide a buildCondaLocal.zsh
script that you can run on your own machine to automatically build an
environment, or you can install the dependencies yourself with conda or pip.

python >= 3.11
    This project uses features that were introduced in Python 3.11 heavily, so
    earlier Python versions will not work.
tfmodisco-lite
    Used to identify motifs.
pyBigWig
    Used to read in data files.
pysam
    used to read in fasta files.
pyBedTools
    Used to read in region files.
tensorflow
    Does the heavy work of machine learning.
tensorflow-probability
    Used to create the multinomial loss function.
matplotlib
    Used to make pretty plots.
tqdm
    Used to create progress bars.
jsonschema
    Used to validate the json files given to the Main CLI programs.

While not strictly necessary, the following packages are very useful:

snakemake
    For automatic processing workflows
jupyterlab
    For interactive data wrangling.
pandoc
    Used to export Jupyter notebooks as pdfs.
pydot, graphviz
    These are only necessary to use showModel.py, which is deprecated and
    will be removed in BPReveal 6.0.0.

Additionally, for development, there are a few libraries that I find useful:

flake8, pydocstyle, pylint
    Used to check the code for style issues.
sphinx, sphinx_rtd_theme, sphinx-argparse, sphinx-autodoc-typehints
    Used to generate the documentation you're reading right now.
coverage
    Used to check code coverage.


