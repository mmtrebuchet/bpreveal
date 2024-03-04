"""Types that are used throughout BPReveal."""
from __future__ import annotations
import typing
import numpy as np
import numpy.typing as npt

ONEHOT_T: typing.TypeAlias = np.uint8
"""Data type for elements of a one-hot encoded sequence."""
ONEHOT_AR_T: typing.TypeAlias = npt.NDArray[ONEHOT_T]
"""Data type for an array of one-hot encoded sequences"""
PRED_T: typing.TypeAlias = np.float32
"""Data type for coverage."""
PRED_AR_T: typing.TypeAlias = npt.NDArray[PRED_T]
"""Data type for an array of predictions."""

LOGIT_T: typing.TypeAlias = np.float32
"""Data type for logits from the model."""
LOGIT_AR_T: typing.TypeAlias = npt.NDArray[LOGIT_T]
"""Data type for an array of logits."""

LOGCOUNT_T: typing.TypeAlias = np.float32
"""Data type for logcount values."""

IMPORTANCE_T: typing.TypeAlias = np.float16
"""Store importance scores with 16 bits of precision.

Since importance scores (particularly PISA values) take up a lot of space, I
use a small floating point type and compression to mitigate the amount of data.
"""

IMPORTANCE_AR_T: typing.TypeAlias = npt.NDArray[IMPORTANCE_T]
"""Data type for an array of importance values."""

MODEL_ONEHOT_T: typing.TypeAlias = np.float32
"""Inside the models, we use floating point numbers to represent one-hot sequences.

For reasons I don't understand, setting this to uint8 DESTROYS pisa values.
"""

MOTIF_FLOAT_T: typing.TypeAlias = np.float32
"""The type used to represent cwms and pwms, and also the type used by the jaccard code.

If you change this, be sure to change libJaccard.c and libJaccard.pyf (and run
make) so that the jaccard library uses the correct data type.
"""
MOTIF_FLOAT_AR_T: typing.TypeAlias = npt.NDArray[MOTIF_FLOAT_T]
"""An array of motif data."""


H5_CHUNK_SIZE: int = 128
"""When saving large hdf5 files, store the data in compressed chunks.

This constant sets the number of entries in each chunk that gets compressed.
For good performance, whenever you read a compressed hdf5 file, it really helps
if you read out whole chunks at a time and buffer them. See, for example,
:py:mod:`shapToBigwig<bpreveal.shapToBigwig>` for an example of a chunked
reader.
"""

QUEUE_TIMEOUT: int = 240  # (seconds)
"""How long should a queue wait before crashing?

In parallel code, if something goes wrong, a queue could stay stuck forever.
Python's queues have a nifty timeout parameter so that they'll crash if they
wait too long. If a queue has been blocking for longer than this timeout, have
the program crash.

This is measured in seconds.
"""


GLOBAL_TENSORFLOW_LOADED: bool = False
"""Has Tensorflow been loaded in this process?

This gets set to True if you use any of the tensorflow-importing functions in
this file. If you import tensorflow in a parent process, child processes will
not be able to use tensorflow, because tensorflow is dumb like that. Tools like
the easy® functions and the threaded batcher check to see if Tensorflow has
been loaded in the parent process before they spawn children.
"""

GENOME_NUCLEOTIDE_FREQUENCY: dict[str, list[float]] = {
    "danRer11": [0.316952, 0.183272, 0.183253, 0.316520],
    "dm6":      [0.290034, 0.210142, 0.209919, 0.289903],  # noqa
    "hg38":     [0.295182, 0.203906, 0.204783, 0.296127],  # noqa
    "mm10":     [0.291497, 0.208327, 0.208343, 0.291831],  # noqa
    "sacCer3":  [0.309806, 0.190882, 0.190596, 0.308714]  # noqa
}
"""The frequency of A, C, G, and T (in that order) in common reference genomes."""


def setTensorflowLoaded():
    """Call this when you first load tensorflow."""
    global GLOBAL_TENSORFLOW_LOADED
    GLOBAL_TENSORFLOW_LOADED = True


def getTensorflowLoaded() -> bool:
    """Returns true if this process has ever loaded tensorflow."""
    return GLOBAL_TENSORFLOW_LOADED
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa  # pylint: disable=line-too-long
