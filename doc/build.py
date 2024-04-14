#!/usr/bin/env python3
"""Builds the .rst files that autodoc will use to generate the documentation."""
import os
import sys
import re


def makeTitle(text: str, borderChar: str, upperBorder: bool = False):
    border = borderChar * len(text)
    if upperBorder:
        fmtStr = "\n{border:s}\n{text:s}\n{border:s}\n\n"
    else:
        fmtStr = "\n{text:s}\n{border:s}\n\n"
    return fmtStr.format(text=text, border=border)


filesText = ["workflow", "programs", "setup", "breakingChanges",
             "modelArchitectures", "countsLossReweighting", "pisa", "bnf"]

filesDevelopment = ["philosophy", "changelog", "releaseChecklist",
                    "license"]

# Things that take a json
filesMajor = ["interpretFlat.py", "interpretPisa.py", "makePredictions.py",
              "motifScan.py", "motifSeqletCutoffs.py", "prepareBed.py",
              "prepareTrainingData.py", "trainCombinedModel.py",
              "trainSoloModel.py", "trainTransformationModel.py"]
# Things that take command-line arguments
filesMinor = ["checkJson.py", "lengthCalc.py", "makeLossPlots.py", "metrics.py",
              "motifAddQuantiles.py", "predictToBigwig.py", "shapToBigwig.py",
              "shapToNumpy.py", "showModel.py", "showTrainingProgress.py"]

# Libraries that can't be executed on their own
filesApi = ["bedUtils.py", "callbacks.py", "gaOptimize.py", "generators.py",
            "interpretUtils.py", "jaccard.py", "layers.py", "logUtils.py",
            "losses.py", "models.py", "motifUtils.py", "schema.py", "training.py",
            "ushuffle.py", "utils.py"]

filesInternalApi = ["disableTensorflowLogging.py", "constants.py", "predictUtils.py"]

filesToolsMinor = ["lossWeights.py", "revcompTools.py", "shiftBigwigs.py",
                   "tileGenome.py", "bestMotifsOnly.py", "shiftPisa.py"]

filesToolsMajor = ["addNoise.py"]

filesToolsApi = ["plots.py", "slurm.py", "addNoiseUtils.py"]

nameModifiers = {
    "tools.": filesToolsApi + filesToolsMajor + filesToolsMinor,
    "internal.": filesInternalApi
}


def formatModuleName(fileName):
    moduleBase = fileName
    if fileName[-3:] == ".py":
        moduleBase = fileName[:-3]
    for modifier, group in nameModifiers.items():
        if fileName in group:
            return modifier + moduleBase
    return moduleBase


def makeHeader():
    with open("_generated/makeHeader", "w") as fp:
        allTargets = []
        for fname in filesMajor + filesMinor + filesApi:
            fp.write("_generated/{0:s}.rst: ../src/{0:s}.py build.py\n\t./build.py {0:s}\n\n".
                     format(fname[:-3]))
            allTargets.append("_generated/{0:s}.rst".format(fname[:-3]))
        for fname in filesToolsApi + filesToolsMinor + filesToolsMajor:
            fp.write("_generated/{0:s}.rst: ../src/tools/{0:s}.py build.py\n\t./build.py {0:s}\n\n".
                     format(fname[:-3]))
            allTargets.append("_generated/{0:s}.rst".format(fname[:-3]))
        for fname in filesInternalApi:
            fp.write(("_generated/{0:s}.rst: ../src/internal/{0:s}.py"
                      " build.py\n\t./build.py {0:s}\n\n").
                     format(fname[:-3]))
            allTargets.append("_generated/{0:s}.rst".format(fname[:-3]))

        for fname in filesText + filesDevelopment:
            fp.write("_generated/{0:s}.rst: text/{0:s}.rst build.py\n\t./build.py {0:s}\n\n".
                     format(fname))
            allTargets.append("_generated/{0:s}.rst".format(fname))
        for fname in filesMajor + filesToolsMajor + ["base.xx", "seqletQuantileCutoffs.xx"]:
            fp.write(
                "_generated/bnf/{0:s}.rst: build.py\n\t./build.py bnf/{0:s}.rst\n\n"
                .format(fname[:-3]))
            allTargets.append("_generated/bnf/{0:s}.rst".format(fname[:-3]))
        for fname in ["_generated/text.rst", "_generated/majorcli.rst",
                      "_generated/minorcli.rst", "_generated/api.rst",
                      "_generated/development.rst", "_generated/toolsapi.rst",
                      "_generated/toolsminor.rst", "_generated/toolsmajor.rst",
                      "_generated/internalapi.rst",
                      "index.rst"]:
            sourceFile = fname.rsplit("/", maxsplit=1)[-1]
            if fname == "index.rst":
                sourceFile = "title.rst"
            fp.write("{0:s}: build.py text/{1:s}\n\t./build.py base\n\n".format(fname, sourceFile))
            allTargets.append("{0:s}".format(fname))
        fp.write("allGenerated = " + " ".join(allTargets) + "\n")


def makeBase():
    ftypes = [["text", "Overview", filesText],
              ["majorcli", "Main CLI", filesMajor],
              ["minorcli", "Utility CLI", filesMinor],
              ["api", "API", filesApi],
              ["toolsminor", "Tools Utility CLI", filesToolsMinor],
              ["toolsmajor", "Tools Main CLI", filesToolsMajor],
              ["toolsapi", "Tools API", filesToolsApi],
              ["internalapi", "Internal", filesInternalApi],
              ["development", "Development", filesDevelopment]]

    for outName, title, contents in ftypes:
        with open(f"_generated/{outName:s}.rst", "w") as fp:
            fp.write(".. Autogenerated by build.py\n")
            fp.write(makeTitle(title, "*", True))
            with open("text/{0:s}.rst".format(outName), "r") as fpIn:
                for line in fpIn:
                    fp.write(line)
            fp.write("\n.. toctree::\n    :maxdepth: 2\n\n")
            for file in contents:
                modName = re.sub(r"(\.py$)|(\.rst$)", "", file)
                fp.write("    {0:s}\n".format(modName))

    # Generate a single .rst index
    with open("index.rst", "w") as fpBig:
        fpBig.write(".. Autogenerated by build.py\n\n")
        fpBig.write(makeTitle("BPReveal Documentataion", "=", True))
        with open("text/title.rst", "r") as inFp:
            for line in inFp:
                fpBig.write(line)
        fpBig.write("\n.. toctree::\n    :maxdepth: 2\n")
        fpBig.write("\n")
        for outName, _, _ in ftypes:
            fpBig.write("    _generated/{0:s}\n".format(outName))

        fpBig.write(makeTitle("Indices", "*", True))
        fpBig.write(
            "* :ref:`genindex`\n* :ref:`modindex`\n* :ref:`search`\n")


def makeBnf(request):
    modRequested = request[4:][:-4]  # strip bnf/ and .rst.
    for fname in filesMajor + filesToolsMajor + ["base.xx", "seqletQuantileCutoffs.xx"]:
        modName = fname[:-3]
        if modRequested == modName:
            inFname = "bnf/{0:s}.bnf".format(modName)
            with open("_generated/bnf/{0:s}.rst".format(modName), "w") as outFp, \
                    open(inFname, "r") as inFp:
                for line in inFp:
                    if m := re.match("[^<]*<([^>]*)> ::=", line):
                        # Create an anchor that I can jump to.
                        outFp.write('.. raw:: html\n\n')
                        outFp.write('    <a name="{0:s}"></a>\n\n'.format(m.group(1)))
                        outFp.write(".. _{0:s}:\n\n".format(m.group(1)))
                        outFp.write(".. highlight:: none\n\n")
                        outFp.write(".. parsed-literal::\n\n")
                        outFp.write("    <:ref:`{0:s}<{0:s}>`> ::=\n".format(m.group(1)))
                    else:
                        outFp.write("    ")
                        inName = False
                        curName = ''
                        for c in line:
                            if not inName:
                                outFp.write(c)
                                if c == '<':
                                    inName = True
                            elif c == '>':
                                outFp.write(
                                    ":ref:`{0:s}<{0:s}>`>".format(curName))
                                inName = False
                                curName = ''
                            else:
                                curName = curName + c


def tryBuildFile(fname):
    modName = fname[:-3]  # Strip off .py
    fmtModName = formatModuleName(fname)
    with open("_generated/" + modName + ".rst", "w") as fp:

        fp.write(".. Autogenerated by build.py\n")
        fp.write(makeTitle(fmtModName, '=', False))

        if fname in filesMajor + filesToolsMajor:
            fp.write(".. automodule:: bpreveal.{0:s}\n    :members:\n\n".
                     format(fmtModName))
            fp.write(".. highlight:: python\n\n")
            fp.write(makeTitle("Schema", "-", False))
            fp.write(".. highlight:: json\n")
            fp.write(".. literalinclude:: ../../src/schematools/{0:s}.schema\n\n".
                     format(modName))

        elif fname in filesMinor + filesToolsMinor:
            fp.write(makeTitle("Help Info", "-", False))
            fp.write(".. highlight:: none\n\n")
            fp.write(".. argparse::\n")
            fp.write("    :module: bpreveal.{0:s}\n".format(fmtModName))
            fp.write("    :func: getParser\n")
            fp.write("    :prog: {0:s}\n\n".format(modName))
            fp.write(makeTitle("Usage", "-", False))
            fp.write("\n.. highlight:: python\n\n")
            fp.write(".. automodule:: bpreveal.{0:s}\n    :members:\n\n".
                     format(fmtModName))
        elif modName == "schema":
            fp.write(".. automodule:: bpreveal.{0:s}\n\n".
                     format(fmtModName))
            fp.write(
                "    .. autodata:: schemaMap(dict[str, Draft7Validator])\n")
            fp.write("        :annotation:\n\n")
            for majorFile in filesMajor + filesToolsMajor:
                fp.write("    .. autodata:: {0:s}(Draft7Validator)\n".format(
                    majorFile[:-3]))
                fp.write("        :annotation:\n\n")

        else:
            fp.write(".. automodule:: bpreveal.{0:s}\n    :members:\n\n".
                     format(fmtModName))
        fp.write("\n.. raw:: latex\n\n    \\clearpage\n")
        fp.write("\n.. raw:: latex\n\n    \\clearpage\n")


def main():
    requestName = sys.argv[1]
    if not os.path.exists("_generated"):
        os.mkdir("_generated")

    if not os.path.exists("_generated/static"):
        os.mkdir("_generated/static")

    if not os.path.exists("_generated/bnf"):
        os.mkdir("_generated/bnf")

    if requestName == "make":
        # In a bit of an incestuous daisy-chain, this program generates
        # a file called makeHeader in _generated, and then
        # the makefile includes it.
        # The makefile also has a rule to make makeHeader, which invokes
        # this script. It's amazing that it works!
        makeHeader()
        return

    if requestName == "base":
        makeBase()
        return

    if requestName.startswith("bnf/"):
        makeBnf(requestName)
        return

    # Generate a .rst file for every module.
    for fname in filesMajor + filesMinor + filesApi + \
            filesToolsMinor + filesToolsApi + filesToolsMajor +\
            filesInternalApi:
        if fname == str(requestName) + ".py":
            tryBuildFile(fname)
            return

    # Now copy over the prose documentation.
    #

    for fname in filesText + filesDevelopment:
        if fname == requestName:
            with open("_generated/{0:s}.rst".format(fname), "w") as fp:
                with open("text/{0:s}.rst".format(fname), "r") as inFp:
                    for line in inFp:
                        fp.write(line)


if __name__ == "__main__":
    main()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa  # pylint: disable=line-too-long
