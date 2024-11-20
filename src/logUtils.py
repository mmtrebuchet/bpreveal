# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Modifications made by Charles McAnany
# - Converted names to camelCase,
# - Changed setVerbosity to take a string and not an int.
# - Removed some functions I don't need.
# - Renamed warn -> warning and fatal -> critical. These are the names used in
#   the Python Standard Library

"""Logging utilities, taken from Tensorflow.

``logUtils`` is meant to be a drop-in replacement for the standard library logging module.
If you're feeling lazy, you can even ``from bpreveal import logUtils as logging``.
"""
import logging as _logging
import sys as _sys
import _thread  # pylint: disable=import-private-name
from collections.abc import Iterable
import traceback as _traceback
from logging import DEBUG
from logging import ERROR
from logging import CRITICAL
from logging import INFO
from logging import WARNING
import threading
import tqdm

# Don't use this directly. Use get_logger() instead.
_BPREVEAL_LOGGER = None
_BPREVEAL_LOGGER_LOCK = threading.Lock()


def _getCaller(offset: int = 3) -> tuple:
    """Returns a code and frame object for the lowest non-logging stack frame."""
    # pylint: disable=protected-access
    f = _sys._getframe(offset)
    # pylint: enable=protected-access
    ourFile = f.f_code.co_filename
    f = f.f_back
    while f:
        code = f.f_code
        if code.co_filename != ourFile:
            return code, f
        f = f.f_back
    return None, None


def _loggerFindCaller(stack_info=False, stacklevel=1):  # pylint: disable=invalid-name # noqa
    del stacklevel
    code, frame = _getCaller(4)
    sinfo = None
    if stack_info:
        sinfo = "\n".join(_traceback.format_stack())
    if code and frame:
        return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
    return "(unknown file)", 0, "(unknown function)", sinfo


def getLogger() -> _logging.Logger:
    """Return TF logger instance.

    :return: An instance of the Python logging library Logger.

    See Python documentation (https://docs.python.org/3/library/logging.html)
    for detailed API. Below is only a summary.

    The logger has 5 levels of logging from the most serious to the least:

    1. CRITICAL
    2. ERROR
    3. WARNING
    4. INFO
    5. DEBUG

    The logger has the following methods, based on these logging levels:

    1. ``critical(msg, *args, **kwargs)``
    2. ``error(msg, *args, **kwargs)``
    3. ``warning(msg, *args, **kwargs)``
    4. ``info(msg, *args, **kwargs)``
    5. ``debug(msg, *args, **kwargs)``

    The `msg` can contain string formatting.  An example of logging at the `ERROR`
    level
    using string formatting is:

    >>> logUtils.getLogger().error("The value %d is invalid.", 3)

    You can also specify the logging verbosity.  In this case, the
    WARNING level log will not be emitted:

    >>> logUtils.setVerbosity("ERROR")
    >>> logUtils.getLogger().warning("This is a warning.")
    """
    global _BPREVEAL_LOGGER

    # Use double-checked locking to avoid taking lock unnecessarily.
    if _BPREVEAL_LOGGER:
        return _BPREVEAL_LOGGER

    with _BPREVEAL_LOGGER_LOCK:
        if _BPREVEAL_LOGGER:
            return _BPREVEAL_LOGGER

        # Scope the TensorFlow logger to not conflict with users' loggers.
        logger = _logging.getLogger("BPReveal")
        logger.propagate = False
        # Override findCaller on the logger to skip internal helper functions
        logger.findCaller = _loggerFindCaller

        # Don't further configure the TensorFlow logger if the root logger is
        # already configured. This prevents double logging in those cases.
        if not logger.handlers:
            # Determine whether we are in an interactive environment
            _interactive = False
            try:
                # This is only defined in interactive shells.
                if _sys.ps1:
                    _interactive = True
            except AttributeError:
                # Even now, we may be in an interactive shell with `python -i`.
                _interactive = _sys.flags.interactive

            # If we are in an interactive environment (like Jupyter), set loglevel
            # to INFO and pipe the output to stdout.
            if _interactive:
                logger.setLevel(INFO)
                _loggingTarget = _sys.stdout
            else:
                _loggingTarget = _sys.stderr

            # Add the output handler.
            loggingFormat = _logging.Formatter("%(levelname)s : %(asctime)s :"
                                               "%(filename)s:%(lineno)d : %(message)s",
                                               datefmt="%Y-%m-%d %H:%M:%S")
            _handler = _logging.StreamHandler(_loggingTarget)
            _handler.setFormatter(loggingFormat)
            logger.addHandler(_handler)

        _BPREVEAL_LOGGER = logger
        return _BPREVEAL_LOGGER


def log(level: int | str, msg: str, *args, **kwargs) -> None:
    """Log message at the given level."""
    if isinstance(level, str):
        level = _LEVEL_MAP[level]
    getLogger().log(level, msg, *args, **kwargs)


def debug(msg: str, *args, **kwargs) -> None:
    """For nitty-gritty details."""
    logger = getLogger()
    logger.debug(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Something went horribly wrong."""
    getLogger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """The world is ending. This is not used by BPReveal."""
    getLogger().critical(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """Normal progress messages."""
    logger = getLogger()
    logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Something the user should pay attention to."""
    getLogger().warning(msg, *args, **kwargs)


_LEVEL_NAMES = {
    CRITICAL: "CRITICAL",
    ERROR: "ERROR",
    WARNING: "WARNING",
    INFO: "INFO",
    DEBUG: "DEBUG",
}

_LEVEL_MAP = {"CRITICAL": _logging.CRITICAL,
              "ERROR": _logging.ERROR,
              "WARNING": _logging.WARNING,
              "INFO": _logging.INFO,
              "DEBUG": _logging.DEBUG}

# Mask to convert integer thread ids to unsigned quantities for logging
# purposes
_THREAD_ID_MASK = 2 * _sys.maxsize + 1

# Counter to keep track of number of log entries per token.
_log_counter_per_token = {}


def logFirstN(level: int, msg: str, n: int, *args: list) -> None:
    """Log 'msg % args' at level 'level' only first 'n' times.

    Not threadsafe.

    :param level: The level at which to log.
    :param msg: The message to be logged.
    :param n: The number of times this should be called before it is logged.
    :param args: The args to be substituted into the msg.
    """
    token = msg % args
    global _log_counter_per_token  # pylint: disable=global-variable-not-assigned
    _log_counter_per_token[token] = 1 + _log_counter_per_token.get(token, -1)
    count = _log_counter_per_token[token]
    logIf(level, msg, count < n, *args)


def logIf(level: str | int, msg: str, condition: bool, *args: list) -> None:
    """Log 'msg % args' at level 'level' only if condition is fulfilled."""
    if condition:
        log(level, msg, *args)


def _getFileAndLine() -> tuple[str, int]:
    """Returns (filename, linenumber) for the stack frame."""
    code, f = _getCaller()
    if not code:
        return ("<unknown>", 0)
    return (code.co_filename, f.f_lineno)  # type: ignore


def getVerbosity() -> int:
    """Return how much logging output will be produced."""
    return getLogger().getEffectiveLevel()


def setVerbosity(userLevel: str | int) -> None:
    """Set the verbosity for this BPReveal session.

    BPReveal uses the python logging module for its printing, and
    less-important information is logged at lower levels.
    Level options are CRITICAL, ERROR, WARNING, INFO, and DEBUG.

    :param userLevel: The level of logging that you'd like to enable.
        It may be an actual logging level (like ``logUtils.ERROR``), or
        a string naming one of the logging levels (like ``"ERROR"``).
    """
    if isinstance(userLevel, str):
        level = _LEVEL_MAP[userLevel]
    else:
        level = userLevel
    getLogger().setLevel(level)
    debug("Logging configured.")


def setBooleanVerbosity(verbose: bool, verboseLevel: str = "INFO",
                        quietLevel: str = "WARNING") -> None:
    """Instead of passing in an int or a string, use a boolean to set verbosity.

    :param verbose: Should the logging be verbose?
    :param verboseLevel: If verbose, how much chatter do you want? Options are the same
        as for :py:func:`setVerbosity<bpreveal.logUtils.setVerbosity>`.
    :param quietLevel: If not verbose, how much chatter do you want? Options are the same
        as for :py:func:`setVerbosity<bpreveal.logUtils.setVerbosity>`.
    """
    if verbose:
        getLogger().setLevel(verboseLevel)
    else:
        getLogger().setLevel(quietLevel)


def _getThreadId() -> int:
    """Get id of current thread, suitable for logging as an unsigned quantity."""
    threadId = _thread.get_ident()
    return threadId & _THREAD_ID_MASK


def wrapTqdm(iterable: Iterable | int, logLevel: str | int = _logging.INFO,
             **tqdmKwargs: dict) -> tqdm.tqdm:
    """Create a tqdm logger or a dummy, based on current logging level.

    :param iterable: The thing to be wrapped, or the number to be counted to.
    :param logLevel: The log level at which you'd like the tqdm to print progress.
    :param tqdmKwargs: Additional keyword arguments passed to tqdm.
    :return: Either a tqdm that will do logging, or an iterable that won't log.
    :rtype: A tqdm-like object supporting either iteration or ``.update()``.

    Sometimes, you want to display a tqdm progress bar only if the logging level is
    high. Call this with something you want to iterate over OR an integer giving the
    total number of things that will be processed (corresponding to::

        pbar = tqdm.tqdm(total=10000)
        while condition:
            pbar.update()
        )

    If iterable is an integer, then this will return a tqdm that you need to
    call update() on, otherwise it'll return something you can use as a loop iterable.

    logLevel may either be a level from the logging module (like logging.INFO) or a
    string naming the log level (like "info")
    """
    if isinstance(logLevel, str):
        # We were given a string, so convert that to a logging level.
        logLevelInternal: int = _LEVEL_MAP[logLevel.upper()]
    elif isinstance(logLevel, int):
        logLevelInternal: int = logLevel
    else:
        raise TypeError(f"Invalid type ({type(logLevel)}) passed to wrapTqdm")

    if isinstance(iterable, int):
        if getLogger().isEnabledFor(logLevelInternal):
            return tqdm.tqdm(total=iterable, **tqdmKwargs)
        return tqdm.tqdm(total=iterable, **tqdmKwargs, disable=True)
    if isinstance(iterable, Iterable):
        iterableOut: Iterable = iterable
        if getLogger().isEnabledFor(logLevelInternal):
            return tqdm.tqdm(iterableOut, **tqdmKwargs)
        return tqdm.tqdm(iterableOut, **tqdmKwargs, disable=True)
    raise TypeError("Your iterable is not valid with tqdm.")
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
