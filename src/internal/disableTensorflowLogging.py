"""A module you can import that does one thing: It makes Tensorflow shut up.

You can import this before you import tensorflow in your code, and some of the warning
messages will go away.

There is also a class (with context manager support) for capturing ALL stderr messages.
Use with caution since it can delete important errors!

"""

import os
import io
import sys
import warnings
import tempfile
import bpreveal.logUtils
from bpreveal.internal.constants import setTensorflowLoaded
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class LeaveStderrAlone:
    """Don't redirect stderr. Can still be used as a context manager.

    This is mostly meant to be used as a dummy when the user has chosen
    not to suppress stderr.
    """

    def __init__(self):
        self.output = None

    def __enter__(self) -> "LeaveStderrAlone":
        """Do nothing."""
        return self

    def __exit__(self, exceptionClass, exceptionValue, traceback) -> None:  # noqa
        """Do nothing."""


class SuppressStderr:
    """Used with a context manager, this will redirect stderr.

    When the context manager leaves, this object will have a string called
    ``output`` that contains all of the garbage that was sent to stderr.

    Note that this will globally redirect stderr, so any logging messages
    will be unceremoniously deleted. Use this class with caution!

    The code for this class is taken from the stream_redirection code in benchopt.
    Benchopt is licensed under the BSD 3-clause license, which is included in
    BPReveal in ``etc/benchopt_license.txt``.
    """

    tempFile: io.BufferedWriter
    origStderrDup: int

    def __init__(self):
        self.origStderrFileno = sys.__stderr__.fileno()
        self.output = None
        self.stderrObj = None

    def __enter__(self) -> "SuppressStderr":
        """Redirect stderr to a temp file."""
        self.origStderrDup = os.dup(self.origStderrFileno)
        self.tempFile = tempfile.TemporaryFile(mode="w+b")
        os.dup2(self.tempFile.fileno(), self.origStderrFileno)
        self.stderrObj = sys.stderr
        sys.stderr = sys.__stderr__
        return self

    def __exit__(self, exceptionClass, exceptionValue, traceback) -> None:  # noqa: ANN001
        """Reset stderr and save the output to self.output."""
        print(flush=True, file=sys.stderr)  # noqa: T201
        sys.stderr = self.stderrObj
        os.close(self.origStderrFileno)
        os.dup2(self.origStderrDup, self.origStderrFileno)
        os.close(self.origStderrDup)
        self.tempFile.flush()
        self.tempFile.seek(0, io.SEEK_SET)
        self.output = self.tempFile.read().decode()
        self.tempFile.close()


_suppressor = SuppressStderr()
try:
    with _suppressor:
        setTensorflowLoaded()
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
except:  # noqa
    bpreveal.logUtils.error("Error importing tensorflow. Error text:")
    bpreveal.logUtils.error(str(_suppressor.output))
    raise
warnings.simplefilter("ignore")
bpreveal.logUtils.debug("Tensorflow logging successfully disabled.")

# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
