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
"""Data type for logits and logcounts."""
PRED_AR_T: typing.TypeAlias = npt.NDArray[PRED_T]
"""Data type for an array of predictions."""

# IMPORTANCE_T: typing.TypeAlias = np.float16
IMPORTANCE_T: typing.TypeAlias = list[str]
"""Store importance scores with 16 bits of precision.

Since importance scores (particularly PISA values) take up a lot of space, I
use a small floating point type and compression to mitigate the amount of data.
"""

MODEL_ONEHOT_T: typing.TypeAlias = np.float32
"""Inside the models, we use floating point numbers to represent one-hot sequences.

For reasons I don't understand, setting this to uint8 DESTROYS pisa values.
"""

MOTIF_FLOAT_T: typing.TypeAlias = np.float32
"""The type used to represent cwms and pwms, and also the type used by the jaccard code.

If you change this, be sure to change libJaccard.c and libJaccard.pyf (and run
make) so that the jaccard library uses the correct data type.
"""


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


def setTensorflowLoaded():
    """Call this when you first load tensorflow."""
    global GLOBAL_TENSORFLOW_LOADED
    GLOBAL_TENSORFLOW_LOADED = True


def getTensorflowLoaded() -> bool:
    """Returns true if this process has ever loaded tensorflow."""
    return GLOBAL_TENSORFLOW_LOADED
