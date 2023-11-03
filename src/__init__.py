"""Utilities for machine learning on genomics data.

See README for a brief description of the programs,
see doc/overview.pdf for a specification,
and see doc/osknExample.ipynb for a demonstration notebook."""


# Version specification:
# Versions are named as [major].[minor].[patch]
# A major version change implies that backwards compatibility will be broken for many uses.
# Data formats on disk may change, and new mandatory parameters may be included in
# configuration files. All previous files and models are invalid after a major version change.

# A minor version change could alter the API and break some code that digs deep into BPReveal.
# Any configuration files will remain valid, and all data files remain valid.
# The minor version can also be incremented when a significant new feature is added.
# If any changes are made to the conda environment (new python version,
# different install procedure, etc.), then the minor version will be incremented.

# A patch change will not break any existing code and does not alter any publically-usable API.
# New minor features and bug fixes will cause patch increments.
# Patch increments will not alter the environment setup, so you can just pull the new BPReveal code
# and it will work with your current conda environment.

__version__ = "3.5.2"
__author__ = "Charles McAnany, Melanie Weilert"
