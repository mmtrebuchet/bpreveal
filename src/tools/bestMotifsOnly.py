#!/usr/bin/env python3
"""Read in a tsv file from motif scanning and remove overlapping motifs.

For each position, if there are two motifs that claim it, remove the motif with a lower
value in the specified score.
"""
import ast
import argparse
from bpreveal import logUtils
from bpreveal.motifAddQuantiles import readTsv, writeTsv
from bpreveal.internal.interpreter import evalAst


def getParser() -> argparse.ArgumentParser:
    """Generates the parser, but does not call parse_args()."""
    p = argparse.ArgumentParser(
        description="Read in a tsv file from the motif scanner and limit each position to "
                    "only have one motif.")
    p.add_argument("--metric",
        help="A python expression giving the value that should be used to compare "
             "motif instances to see which one is better. (Beds should use the score "
             "column to compare.)",
        default="score")
    p.add_argument("--filter",
        help="Only consider motifs that have at least this value in the given column. "
                   "Format: Any valid Python expression where the identifiers are "
                   "column names in the tsv. "
                   "(Don't forget to quote comparison operators on the shell!)",
        default="True",
        dest="filter",
        type=str)
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
    p.add_argument("--no-match-names",
        help="If provided, then don't require motif names to match. "
             "Default: motifs will only be removed if there is a better "
             "instance of a motif with the same name at that locus.",
        dest="matchNames",
        action="store_false")
    p.add_argument("--max-offset",
        help="Instead of removing motifs that overlap at all, only compare "
             "motif instances that are offset by this amount or less.",
        dest="maxOffset",
        type=int,
        default=99999)
    p.add_argument("--verbose",
        help="Show progress.",
        action="store_true")
    return p


def removeOverlaps(entries: list[dict], metric: ast.AST, nameCol: str | None,
                   maxOffset: int) -> list[dict]:
    """Scan over the (sorted) motif hits and keep the best ones.

    :param entries: A list of motif tsv (or bed) entries. This is a dict keyed
        by column name.
    :param metric: The Python expression used to make comparisons, like ``score``.
    :param nameCol: The name of the column used to check to see if motifs have the
        same name, for example ``short_name``. If ``None``, then don't compare
        motifs by name, and only return the single best hit at each locus.
    :param maxOffset: If two entries have midpoints that are separated by more than
        this distance, then they are considered non-overlapping.
        The larger this value, the more aggressive the culling will be.
    :return: A list of entries, of the same type as the input entries.
    """
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
        myMetric = evalAst(metric, e)
        assert isinstance(myMetric, (int, float))
        myMidpoint = (e["end"] + e["start"]) / 2
        for j in range(scanStart, scanEnd + 1):
            if i != j:
                other = entries[j]
                theirMidpoint = (other["end"] + other["start"]) / 2
                if abs(theirMidpoint - myMidpoint) > maxOffset:
                    # They overlap, but by less than maxOffset.
                    continue
                if nameCol is not None:
                    # Only compare to records with the same name.
                    if other[nameCol] != e[nameCol]:
                        continue

                theirMetric = evalAst(metric, other)
                assert isinstance(theirMetric, (int, float))
                if theirMetric < myMetric:
                    recordEntry = False
                    break
                elif e[metric] == other[metric]:
                    # We have a tie. I need to pick
                    # a winner, so I'll say the motif on the left wins.
                    other[metric] = other[metric] - 10000
        if recordEntry:
            outEntries.append(e)
    return outEntries


def main() -> None:
    """Run the culling algorithm."""
    args = getParser().parse_args()
    logUtils.setBooleanVerbosity(args.verbose)
    nameCol = None
    if args.inTsv is not None:
        inFname = args.inTsv
        colNames, entries, _ = readTsv(inFname)
        if args.matchNames:
            nameCol = "short_name"
    else:
        assert args.inBed is not None, "Must provide tsv or bed!"
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
        if args.matchNames:
            nameCol = "name"

    def sortKey(e: dict) -> tuple[str, int]:
        return (e["chrom"], e["start"])
    logUtils.info("Loaded input file")
    sortEntries = sorted(entries, key=sortKey)
    strippedEntries = []
    filterAst = ast.parse(args.filter)
    for e in sortEntries:
        if evalAst(filterAst, e):
            strippedEntries.append(e)
    sortEntries = strippedEntries
    metricAst = ast.parse(args.metric)
    outs = removeOverlaps(sortEntries, metricAst, nameCol, args.maxOffset)
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
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
