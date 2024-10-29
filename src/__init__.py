"""Utilities for machine learning on genomics data.

See README for a brief description of the programs,
See the documentation (which you can build) for a specification,
and see doc/demos/osknExample.ipynb for a demonstration notebook.
"""


# Version specification:
# Versions are named as [major].[minor].[patch]
# A major version change implies that backwards compatibility will be broken for many
# uses. Data formats on disk may change, and new mandatory parameters may be included in
# configuration files. All previous files and models are invalid after a major version
# change.

# A minor version change could alter the API and break some code that digs deep into
# BPReveal. Any configuration files will remain valid, and all data files remain valid.
# The minor version can also be incremented when a significant new feature is added.
# If any changes are made to the conda environment (new python version,
# different install procedure, etc.), then the minor version will be incremented.

# A patch change will not break any existing code and does not alter any publicly-usable
# API. New minor features and bug fixes will cause patch increments. Patch increments
# will not alter the environment setup, so you can just pull the new BPReveal code
# and it will work with your current conda environment.

__version__ = "5.0.1"
__author__ = "Charles McAnany, Ph.D.; Melanie Weilert; Haining Jiang; "\
             "Patrick Moeller; Samuel Campbell; Anshul Kundaje, Ph.D.; "\
             "Julia Zeitlinger, Ph.D."
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
