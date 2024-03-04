#!/usr/bin/env python3
"""Read in a tsv file from motif scanning and remove overlapping motifs.

For each position, if there are two motifs that claim it, remove the motif with a lower
value in the specified score.
"""

import argparse
from bpreveal import logUtils
from bpreveal.motifAddQuantiles import readTsv, writeTsv


def getParser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Read in a tsv file from the motif scanner and limit each position to "
                    "only have one motif.")
    p.add_argument("--column",
        help="The name of the tsv column that should be used to compare two called motifs "
             "to see which one is better. (Beds should use the score column to compare.)",
        default="score")
    p.add_argument("--in-tsv",
        help="The input file, in TSV format. One of --input-tsv or --input-bed is required",
        dest="inTsv")
    p.add_argument("--in-bed",
        help="The input file, in bed format. One of --input-tsv or --input-bed is required",
        dest="inBed")
    p.add_argument("--out-bed",
        help="(Optional) The name of the output file, in bed format.",
        dest="outBed")
    p.add_argument("--out-tsv",
        help="(Optional) The name of the output file, in tsv format. "
             "Cannot be used with --in-bed.",
        dest="outTsv")
    p.add_argument("--verbose",
        help="Show progress.",
        action="store_true")
    return p


def removeOverlaps(entries, colName):
    outEntries = []
    for i, e in logUtils.wrapTqdm(enumerate(entries), total=len(entries)):
        scanStart = i
        while scanStart > 0 \
                and entries[scanStart - 1]["end"] >= e["start"] \
                and entries[scanStart - 1]["chrom"] == e["chrom"]:
            scanStart -= 1
        scanEnd = i
        while scanEnd < len(entries) - 1 \
                and entries[scanEnd + 1]["start"] <= e["end"] \
                and entries[scanEnd + 1]["chrom"] == e["chrom"]:
            scanEnd += 1
        recordEntry = True
        for j in range(scanStart, scanEnd + 1):
            if i != j:
                other = entries[j]
                if e[colName] < other[colName]:
                    recordEntry = False
                    break
        if recordEntry:
            outEntries.append(e)
    return outEntries


def main():
    args = getParser().parse_args()
    logUtils.setBooleanVerbosity(args.verbose)
    if args.inTsv is not None:
        inFname = args.inTsv
        colNames, entries, _ = readTsv(inFname)
    elif args.inBed is not None:
        inFname = args.inBed
        colNames = ["chrom", "start", "end", "name", "score", "strand"]
        convFns = [str, int, int, str, float, str]
        entries = []
        with open(inFname, "r") as fp:
            for line in fp:
                curEntry = {}
                lsp = line.split()
                for i, elem in enumerate(lsp):
                    curEntry[colNames[i]] = convFns[i](elem)
                entries.append(curEntry)

    def sortKey(e):
        return (e["chrom"], e["start"])
    logUtils.info("Loaded input file")
    sortEntries = sorted(entries, key=sortKey)
    outs = removeOverlaps(sortEntries, args.column)
    if args.outTsv is not None:
        writeTsv(outs, colNames, args.outTsv)
    if args.outBed is not None:
        with open(args.outBed, "w") as fp:
            for o in outs:
                outElems = []
                for cn in colNames[:6]:
                    if cn in o:
                        outElems.append(str(o[cn]))
                outLine = "\t".join(outElems)
                fp.write(outLine + "\n")
    logUtils.info("Motif curation complete.")


if __name__ == "__main__":
    main()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa  # pylint: disable=line-too-long
