#!/usr/bin/env python3
"""Read in a tsv file from motif scanning and remove overlapping motifs.

For each position, if there are two motifs that claim it, remove the motif with a lower
value in the specified score.
"""
import ast
import argparse
from typing import Any
from bpreveal import logUtils
from bpreveal.motifAddQuantiles import readTsv, writeTsv


def getParser() -> argparse.ArgumentParser:
    """Generates the parser, but does not call parse_args()."""
    p = argparse.ArgumentParser(
        description="Read in a tsv file from the motif scanner and limit each position to "
                    "only have one motif.")
    p.add_argument("--column",
        help="The name of the tsv column that should be used to compare two called motifs "
             "to see which one is better. (Beds should use the score column to compare.)",
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
    p.add_argument("--match-names",
        help="If provided, then only compare motifs with the same name. "
             "Overlapping motifs will only be removed if there is a better "
             "instance of a motif with the same name at that locus.",
        dest="matchNames",
        action="store_true")
    p.add_argument("--verbose",
        help="Show progress.",
        action="store_true")
    return p


def evalAst(t: ast.AST, env: dict[str, Any]) -> int | float | bool:
    """Evaluates the (ast.parse()d) filter string t using variables in the environment env.

    :param t: The parsed AST that should be evaluated
    :param env: The environment containing the variables used in the expression.
    :return: The value of the expression.


    Syntax:

    This interpreter interprets a subset of the Python programming language.
    Since it uses the Python parser, it obeys Python's operator precedence.
    If it encounters a name in the expression, it looks it up in the supplied environment.

    .. highlight:: none

    .. literalinclude:: ../../doc/bnf/bestMotifsOnly.bnf

    """
    match t:
        case ast.Module():
            return evalAst(t.body[0], env)
        case ast.Constant():
            return t.value
        case ast.Expr():
            return evalAst(t.value, env)
        case ast.Compare():
            prev = evalAst(t.left, env)
            for op, rhs in zip(t.ops, t.comparators):
                rv = evalAst(rhs, env)
                match op:
                    case ast.Lt():
                        if prev >= rv:
                            return False
                    case ast.Gt():
                        if prev <= rv:
                            return False
                    case ast.LtE():
                        if prev > rv:
                            return False
                    case ast.GtE():
                        if prev < rv:
                            return False
                    case ast.NotEq():
                        if prev == rv:
                            return False
                    case ast.Eq():
                        if prev != rv:
                            return False
                    case _:
                        raise SyntaxError(f"Unsupported comparison operator in expression: {op}")
                # Go to next comparison.
                prev = rv
            return True
        case ast.BinOp():
            lhs = evalAst(t.left, env)
            rhs = evalAst(t.right, env)
            match t.op:
                case ast.Add():
                    return lhs + rhs
                case ast.Sub():
                    return lhs - rhs
                case ast.Mult():
                    return lhs * rhs
                case ast.Div():
                    return lhs / rhs
                case _:
                    raise SyntaxError(f"Unsupported binary operator in expression: {t.op}")
        case ast.BoolOp():
            match t.op:
                case ast.And():
                    for v in t.values:
                        if not evalAst(v, env):
                            return False
                    return True
                case ast.Or():
                    for v in t.values:
                        if evalAst(v, env):
                            return True
                    return False
                case _:
                    raise SyntaxError(f"Unsupported boolean operator in expression: {t.op}")
        case ast.UnaryOp():
            match t.op:
                case ast.USub():
                    return -evalAst(t.operand, env)
                case ast.Not():
                    return not evalAst(t.operand, env)
                case _:
                    raise SyntaxError(f"Unsupported unary operator in expression: {t.op}")
        case ast.Name():
            return env[t.id]
        case _:
            print(ast.dump(t))
            print("no")
            return "error"  # type: ignore


def removeOverlaps(entries: list[dict], colName: str, nameCol: str | None) -> list[dict]:
    """Scan over the (sorted) motif hits and keep the best ones.

    :param entries: A list of motif tsv (or bed) entries. This is a dict keyed
        by column name.
    :param colName: The name of the column used to make comparisons, like ``score``.
    :param nameCol: The name of the column used to check to see if motifs have the
        same name, for example ``short_name``. If ``None``, then don't compare
        motifs by name, and only return the single best hit at each locus.
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
        for j in range(scanStart, scanEnd + 1):
            if i != j:
                other = entries[j]
                if nameCol is not None:
                    # Only compare to records with the same name.
                    if other[nameCol] != e[nameCol]:
                        continue
                if e[colName] < other[colName]:
                    recordEntry = False
                    break
                elif e[colName] == other[colName]:
                    # We have a tie. I need to pick
                    # a winner, so I'll say the motif on the left wins.
                    other[colName] = other[colName] - 10000
        if recordEntry:
            outEntries.append(e)
    return outEntries


def main():
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

    outs = removeOverlaps(sortEntries, args.column, nameCol)
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
