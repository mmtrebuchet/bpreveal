"""A wrapper around the ushuffle C implementation."""
import threading
import numpy as np
from bpreveal.internal import libushuffle
from bpreveal.internal.constants import ONEHOT_AR_T

# The ushuffle implementation in C makes heavy use of global variables.
# To avoid multiple threads trampling over each other and causing races,
# declare a lock that must be obtained before ushuffle code is run.
# Note that this is a THREADING lock, not a MULTIPROCESSING lock.
# With multiprocessing, each process has its own copy of the global variables
# in ushuffle, so there's no need for a lock between processes.
_SHUFFLE_LOCK = threading.Lock()

# Set up the RNG.
libushuffle.initialize()


def shuffleString(sequence: str, kmerSize: int, numShuffles: int = 1,
                  seed: int | None = None) -> list[str]:
    """Given a string sequence, perform a shuffle that maintains the kmer distribution.

    This is adapted from ushuffle.
    ``sequence`` should be a string in ASCII, but it should theoretically work
    on multi-byte encoded utf-8 characters so long as the kmerSize is at least
    as long as the longest byte sequence for a character in the input.
    (Please don't rely on this random fact!)

    Returns a list of shuffled strings.
    """
    ar = np.frombuffer(sequence.encode("utf-8"), dtype=np.int8)
    with _SHUFFLE_LOCK:
        if seed is not None:
            libushuffle.seedRng(seed)
        shuffledArrays = libushuffle.shuffleStr(ar, kmerSize, numShuffles)
    ret = []
    for i in range(numShuffles):
        ret.append(shuffledArrays[i].tobytes().decode("utf-8"))
    return ret


def shuffleOHE(sequence: ONEHOT_AR_T, kmerSize: int, numShuffles: int = 1,
               seed: int | None = None) -> ONEHOT_AR_T:
    """Given a one-hot sequence, perform a shuffle that maintains the kmer distribution.

    ``sequence`` should have shape ``(length, alphabetLength)``. For DNA, ``alphabetLength == 4``.
    It is an error to have an alphabet length of more than 8. Internally,
    this function packs the bits at each position into a character, and the
    resulting string is shuffled and then unpacked. For this reason, it is
    possible to have more than one letter be hot at one position, or even to
    have no letters hot at a position. For example, this one-hot encoded
    sequence is valid input::

        Pos A C G T
        0   1 0 0 0
        1   0 1 0 0
        2   1 0 1 0
        3   0 1 1 1
        4   0 0 0 0

    This is adapted from ushuffle.
    Returns an array of shape ``(numShuffles, length, alphabetLength)``
    """
    assert sequence.shape[1] <= 8, "Cannot ushuffle a one-hot encoded sequence with "\
                                   "an alphabet of more than 8 characters."
    with _SHUFFLE_LOCK:
        if seed is not None:
            libushuffle.seedRng(seed)
        shuffledSeqs = libushuffle.shuffleOhe(sequence, kmerSize, numShuffles)
    return shuffledSeqs
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
