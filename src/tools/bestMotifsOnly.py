#!/usr/bin/env python3
"""Read in a tsv file from motif scanning and remove overlapping motifs.

For each position, if there are two motifs that claim it, remove the motif with a lower
value in the specified score.

The ``metric`` and ``filter`` arguments are evaluated by the interpreter.
These expressions are evaluated in an environment where each property of a record
(either bed or tsv) is bound to a variable given by the column name. For example, if a
bed file is provided, then ``chrom``, ``name``, ``start``, and the other columns are
variables in the environment for the expression. You can see the
:py:mod:`interpreter<bpreveal.internal.interpreter>` documentation for a list of all
of the syntax available, but here are a few examples:

``--metric score``
    would rank motifs based on their score column, assuming bed-format input.
``--metric 'seq_match_quantile + 2 * contrib_match_quantile'``
    would score motifs based mostly on their contribution match, but also
    include some weight for sequence match.
``--filter 'contrib_match_quantile > 0.8'``
    would only keep the top 20 percent of motifs no matter the motif name.
``--filter '(pattern_name != "polyA") or (contrib_magnitude_quantile > 0.9)'``
    would select all motifs not named ``polyA`` and only accept the top 10 percent of
    ``polyA`` motifs.

``filter`` should return True or False, whereas ``metric`` should return a scalar.

Note that there is no support for comparing motifs as part of your metric, so there
is no way to say
``--metric 'motif1.contrib_magnitude if motif1.end > motif2.start else motif1.contrib_match'``.
(The names ``motif1`` and ``motif2`` would be name errors, since they are not in scope.)

"""
import ast
import argparse
from bpreveal import logUtils
from bpreveal.motifAddQuantiles import readTsv, writeTsv
from bpreveal.internal.interpreter import evalAst


def getParser() -> argparse.ArgumentParser:
    """Generate the parser, but do not call parse_args()."""
    p = argparse.ArgumentParser(
        description="Read in a tsv file from the motif scanner and limit each position to "
                    "only have one motif.")
    p.add_argument("--metric",
        help="A python expression giving the value that should be used to compare "
             "motif instances to see which one is better. (Beds should use the 'score' "
             "column to compare.) See man page for examples.",
        default="score")
    p.add_argument("--filter",
        help="Only consider motifs that satisfy this filter. "
             "Format: Any valid Python expression where the identifiers are "
             "column names in the tsv. "
             "(Don't forget to quote comparison operators on the shell!) "
             "See the man page for examples.",
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


def removeOverlaps(entries: list[dict], nameCol: str | None,
                   maxOffset: int) -> list[dict]:
    """Scan over the (sorted) motif hits and keep the best ones.

    For every pair of motifs (m1, m2):

    1. If they don't overlap at all, continue.
    2. If the center of m1 is more than ``maxOffset`` bp away from the center
       of m2, continue.
    3. If nameCol is given (i.e., is not none) and the name of m1 is different
       than the name of m2, continue.
    4. We have established that m1 and m2 are competing. We must mark one of
       them as bad.
    5. Select the motif that has the lower metric. Mark it as bad.
       If they have the same metric value, mark one at random.

    Once all of the marking is complete, return a list of all motifs that
    are NOT marked as bad.

    :param entries: A list of motif tsv (or bed) entries. This is a dict keyed
        by column name.
    :param nameCol: The name of the column used to check to see if motifs have the
        same name, for example ``short_name``. If ``None``, then don't compare
        motifs by name, and only return the single best hit at each locus.
    :param maxOffset: If two entries have midpoints that are separated by more than
        this distance, then they are considered non-overlapping.
        The larger this value, the more aggressive the culling will be.
    :return: A list of entries, of the same type as the input entries.

    There are some quirks with this algorithm. It works based on the order of the
    bed file, and so the following situation gives a result you might not expect:

    .. highlight:: none

    ::

        |---a:0.8---|     |---c:0.2---|
                 |---b:0.7---|   |---d:0.1---|

    Here we have four motif calls: a, b, c, and d. By the above algorithm,
    the motif pairs (a, b), (b, c), and (c, d) have overlap, and so we'll mark
    the lower-scoring motif from each pair.
    Between a and b, b has the lower score, so b will be marked.
    For b and c, c will be marked.
    For c and d, d will be marked.
    This means that ONLY motif a will be returned because every other motif instance
    had overlap with a better instance.
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

                if other["metric"] > e["metric"]:
                    recordEntry = False
                    break
                if e["metric"] == other["metric"]:
                    # We have a tie. I need to pick
                    # a winner, so I'll say the motif on the left wins.
                    other["metric"] = other["metric"] - 10000
        if recordEntry:
            outEntries.append(e)
    return outEntries


def loadEntries(inTsv: str | None, inBed: str | None,
                matchNames: bool) -> tuple[list[dict], list[str], str | None]:
    """Read in the bed or TSV file containing motif calls.

    :param inTsv: The name of the input tsv file, or None if one wasn't provided.
    :param inBed: The name of the input bed file, or None if one wasn't provided.
    :param matchNames: Should the culling only consider motifs that have the same name?
    :return: A tuple containing three items. The first is a list of dicts,
        containing all of the entries from the input data file. The second is a
        list of the field names in each entry. The third is a string giving the
        name of the field that should be used to get the name of the motif.
    """
    nameCol = None
    if inTsv is not None:
        assert inBed is None, "Cannot specify both a tsv and bed input. Choose one."
        inFname = inTsv
        colNames, entries, _ = readTsv(inFname)
        if matchNames:
            nameCol = "short_name"
    else:
        assert inBed is not None, "Must provide tsv or bed!"
        inFname = inBed
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
        if matchNames:
            nameCol = "name"
    return entries, colNames, nameCol


def preprocessMotifs(entries: list[dict], filterDef: str, metricDef: str) -> list[dict]:
    """Sort, filter, and score the given motif hits.

    :param entries: A list of dicts, each one representing one motif hit and
        the fields corresponding to the columns in the data file.
    :param filterDef: A string giving the filter to apply to each of the motifs.
    :param metricDef: A string giving the metric to be calculated for scoring.
    :return: A list of entries that pass the filter. Each one will have a new field
        called ``metric`` that contains the calculated metric for that motif instance.
    """
    def sortKey(e: dict) -> tuple[str, int]:
        return (e["chrom"], e["start"])
    sortEntries = sorted(entries, key=sortKey)
    strippedEntries = []
    filterAst = ast.parse(filterDef)
    for e in sortEntries:
        if evalAst(filterAst, e, True):
            strippedEntries.append(e)
    logUtils.info(f"Filtering complete. Surviving motifs: {len(strippedEntries)}")
    metricAst = ast.parse(metricDef)
    measuredEntries = []
    for e in strippedEntries:
        # We don't need to add functions to the entries again, since we did that during filter.
        e["metric"] = evalAst(metricAst, e, True)
        measuredEntries.append(e)
    return measuredEntries


def main() -> None:
    """Run the culling algorithm."""
    args = getParser().parse_args()
    logUtils.setBooleanVerbosity(args.verbose)
    entries, colNames, nameCol = loadEntries(args.inTsv, args.inBed, args.matchNames)

    logUtils.info(f"Loaded input file. Number of entries: {len(entries)}")
    measuredEntries = preprocessMotifs(entries, args.filter, args.metric)
    logUtils.info("Metrics calculated. Beginning overlap removal calculation.")
    outs = removeOverlaps(measuredEntries, nameCol, args.maxOffset)
    logUtils.info(f"Culling complete. Surviving motifs: {len(outs)}")
    if args.outTsv is not None:
        logUtils.info("Writing output tsv.")
        writeTsv(outs, colNames, args.outTsv)
    if args.outBed is not None:
        logUtils.info("Writing output bed.")
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
