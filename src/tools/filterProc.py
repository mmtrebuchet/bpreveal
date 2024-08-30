#!/usr/bin/env python3
"""A little script that removes unhelpful warnings from program outputs.

Can be used either in a pipe, as in ``cat log.out | filterProc``
or as a parent process as in ``filterProc trainSoloModel config.json``.
In the second case, this program will capture both stdout and stderr.

"""
import os
import argparse
import subprocess as sp
import selectors
import re
import sys

_badLineStrs = [
    r"^W0000.*Skipping the delay kernel.*",
    r"^WARNING: All log messages before absl::InitializeLog\(\) is called are written to STDERR$",
    r"^I0000.*XLA service.* initialized for platform CUDA",
    r"^I0000.*StreamExecutor device \(0\):",
    r"^I0000.*Compiled cluster using XLA!",
    r"^I0000.*successful NUMA node read from SysFS had negative value.*$",
]

_normalLineStrs = [
    r"^DEBUG : ",
    r"^INFO : ",
    r".*         $",
    r"^\|.*   \|$",
    r"^_*$",
    r"^.-*.$",
    r"^.=*.$",
    r"^(Total|Trainable|Non-trainable) params.*$",
    r"^reference.*predicted.*regions.*$",
    r"^(metric |mnll |jsd |pearsonr |spearmanr |Counts pearson |Counts spearman).*$",
    r"^.*Gen [0-9]* fitness.*$",
    r"^Epoch [0-9]*: val_loss .*$",
    r"^Epoch [0-9]*: early stopping$",
    r"^Epoch [0-9]*: ReduceLROnPlateau reducing learning rate to.*$",
    r"^Restoring model weights from the end of the best epoch",
]

_badLineRegexes = [re.compile(x) for x in _badLineStrs]
_normalLineRegexes = [re.compile(x) for x in _normalLineStrs]


class Writer:
    """Accepts data from pipes and writes it to stdout once it encounters a newline."""

    def __init__(self, quiet):
        self.buf = []
        self.quiet = quiet

    def add(self, data):
        """Add a bytes object of output from a pipe."""
        for c in data:
            self.buf.append(c)
            if c in {ord("\r"), ord("\n")}:
                self.printLine()

    def printLine(self):
        """Once an end of line has been hit, print the current line."""
        outar = bytes(self.buf).decode("utf-8")
        if self.checkLine(outar):
            print(outar, end="")
        self.buf = []

    def checkLine(self, line):
        """Should the given line be printed? Checks against the regexes."""
        for r in _badLineRegexes:
            m = r.match(line)
            if m is not None:
                return False
        if self.quiet:
            for r in _normalLineRegexes:
                m = r.match(line)
                if m is not None:
                    return False
        return True


def runProc(command, quiet: bool):
    """Lets the process run and scrapes its stdout and stderr.

    :param command: An array of strings giving the command to run.
    :param quiet: Should this filter also suppress normal informational messages?
    """
    proc = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ, data="child")  # type: ignore
    sel.register(proc.stderr, selectors.EVENT_READ, data="child")  # type: ignore
    sel.register(sys.stdin, selectors.EVENT_READ, data="input")  # type: ignore
    outBuf = Writer(quiet)
    errBuf = Writer(quiet)
    inBuf = Writer(quiet)
    moreToDo = True
    while moreToDo:
        selhits = sel.select()
        for key, _ in selhits:
            if key.data == "child":
                data = key.fileobj.read1()  # type: ignore
                if not data:
                    moreToDo = False
                elif key.fileobj is proc.stdout:
                    outBuf.add(data)
                elif key.fileobj is proc.stderr:
                    errBuf.add(data)
            else:
                # Only read a single character at a time. Slow, but avoids
                # all blocking problems.
                rv = key.fileobj.read(1)  # type: ignore
                if not rv:
                    moreToDo = False
                    continue
                data = bytes(rv, encoding="utf-8")
                inBuf.add(data)
    if not sys.stdin.isatty():
        for line in sys.stdin:
            inBuf.add(bytes(line, encoding="utf-8"))

def getParser():
    """Build (but don't parse_args) the parser."""
    ap = argparse.ArgumentParser(description="Run the output from the given command "
        "through a few filters that will strip out unnecessary warnings.")
    ap.add_argument("--quiet", help="Suppress info and debug messages, leaving only "
                    "serious warnings.", action="store_true")
    ap.add_argument("command", nargs="*", help="The command to run")
    return ap


def main():
    """Run the program."""
    ap = getParser()
    args = ap.parse_args()
    if len(args.command) == 0:
        args.command = ["true"]
    runProc(args.command, args.quiet)

if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "True"
    main()
