#!/usr/bin/env python3
r"""Calculates quantile values for seqlets and called motif instances.

This little helper program calculates quantile values for seqlets and called
motif instances. For each pattern (patterns in different metaclusters are
distinct), it looks at the seqlets and determines where that seqlet's
importance magnitude, contribution match, and sequence match scores fall among
other seqlets in that pattern. Then, for motif hits, it sees where each hit
falls, in terms of quantile, among the seqlets in that pattern.

The quantile is based on a very simple definition. A particular seqlet's
quantile is calculated by sorting all of the seqlets for one pattern. The
lowest-scoring seqlet gets a quantile of 0.0, the highest-scoring gets 1.0, and
the seqlets in between get quantile values based on their order in the sorted
metric.

For scanned hits, we take the sorted array of seqlet statistics (for the same
pattern as the matched hit fell into) and ask, 'where would the score of this
hit rank among the sorted array of seqlet scores?' The hit's rank is then its
quantile score. If a hit falls between the scores of two seqlets, then a linear
interpolation is performed to assign a quantile value.

The input and output to this program are tsv files, and the only difference in
format is that the outputs have three additional columns:
``contrib_magnitude_quantile``, ``seq_match_quantile``, and
``contrib_match_quantile``. The remaining columns are described below:

chrom
    The chromosome where the seqlet or hit was found.

start
    The start position (inclusive, zero-based) of the seqlet or motif hit.

end
    The end position (exclusive, zero-based) of the seqlet or motif hit.

short_name
    The user-provided name for this motif. If you didn't provide one in the
    configuration for
    :py:mod:`motifSeqletCutoffs<bpreveal.motifSeqletCutoffs>`, then it will be
    something like ``pos_0`` for the positive metacluster, pattern zero.

contrib_magnitude
    The total contribution across this motif instance. A higher value means
    more motif contribution.

strand
    A single character indicating if the motif was on the positive or negative
    strand.

metacluster_name
    Straight from the modisco hdf5 file. It will be something like
    ``pos_patterns``.

pattern_name
    Also from the modisco hdf5. It will be something like ``pattern_5``.

sequence
    The DNA sequence of that motif instance.

index
    Either the region index in the contribution hdf5 (from
    :py:mod:`motifScan<bpreveal.motifScan>`), or the seqlet index in the
    modisco hdf5 (from
    :py:mod:`motifSeqletCutoffs<bpreveal.motifSeqletCutoffs>`).

seq_match
    The information content of the sequence
    match to the motif's pwm.

contrib_match
    The continuous Jaccard similarity
    between the motif's cwm and the contribution scores of this seqlet.

seq_match_quantile
    Given the PSSM score of each mapped hit to the original TF-MoDISco PSSM,
    calculate the quantile value of this score, given the distribution of
    seqlets corresponding to the TF-MoDISco pattern.

contrib_match_quantile
    Given the CWM score (i.e. Jaccardian-similarity) of each mapped hit's
    contribution to the original TF-MoDISco CWM, calculate the quantile value
    of this score, given the distribution of seqlets corresponding to the
    TF-MoDISco pattern.

contrib_magnitude_quantile
    Given the total L1 magnitude of contribution across a mapped hit, calculate
    the quantile value of this magnitude, given the distribution of seqlets
    corresponding to theTF-MoDISco pattern.

If a contribution hdf5 file was not provided to
:py:mod:`motifSeqletCutoffs<bpreveal.motifSeqletCutoffs>`, the chrom, start,
and end columns are meaningless.

Additional Information
----------------------

Converting to bed
^^^^^^^^^^^^^^^^^
The first six columns define a bed
file, and a simple ``cut`` command can generate a viewable bed file from these
tsvs::

    cat scan.tsv | cut -f 1-6 | tail -n +2 > scan.bed

Removing duplicates
^^^^^^^^^^^^^^^^^^^
The hits from scanning can contain duplicates. This can happen if the
same bases appear in multiple regions (i.e., there is overlap in the region
set). In this case, it makes sense to only keep the best instance (highest
importance magnitude) of that motif hit. This can be done with a little
Unix-fu::

    cat scan.tsv | \
        cut -f 1-6 | \
        tail -n +2 | \
        sort -k1,1 -k2,2n -k3,3n -k4,4 -k5,5nr | \
        awk '!_[$1,$2,$3,$4,$6]++' > scan.bed

For a more general but still somewhat user-friendly tool to remove duplicates,
see the :py:mod:`bestMotifsOnly<bpreveal.tools.bestMotifsOnly>` tool.

API
---

"""
# flake8: noqa: ANN
import csv
import argparse
import numpy as np
from bpreveal import motifUtils
from bpreveal import logUtils

def recordToPatternID(record):
    """Come up with an identifier that uniquely identifies a particular pattern.

    Since a single csv should only represent one modisco run, it's safe to just
    mash the metacluster and pattern names together. We can't use short-name because
    somebody could give the same name to different patterns.
    """
    mn = record["metacluster_name"].split("_")[0]
    pn = record["pattern_name"].split("_")[1]
    return mn + pn

def numericalize(row: dict[str, str]) -> dict[str, float | int | str]:
    """Given a row from a TSV DictReader, parse any numbers you find.

    :param row: A dict from a TSV file containing numbers or strings.
    :return: A new dict where numerical values are actually number types in Python
    """
    ret = {}
    for k, v in row.items():
        try:
            ret[k] = int(v)
        except ValueError:
            try:
                ret[k] = float(v)
            except ValueError:
                ret[k] = v
    return ret


def readTsv(fname: str):
    """Reads in a tsv file generated by the motif seqlet cutoff and scanning tools.

    Returns a three-tuple. The first contains the field names from the tsv file, in order.
    The second is a list of dicts, each dict corresponding to one field in the tsv.
    The dicts map field names (strings) to the contents of the corresponding column
    for that record. The fields are numbers if possible, because they've been run through
    :py:func:`~numericalize`.
    The third value returned is a list of the unique pattern identifiers among the
    records; these names are generated by recordToPatternID
    Each record in the returned list of records contains a field that was not present in
    the initial tsv, this field is called _TMPNAME. This field contains the combined
    pattern identifier.
    """
    records = []
    patternIDs = []
    with open(fname, "r", newline="") as fp:
        fieldNames = [x.strip() for x in fp.readline().split("\t")]
        reader = csv.DictReader(fp, fieldnames=fieldNames, delimiter="\t")
        for row in reader:
            tmpName = recordToPatternID(row)
            row["_TMPNAME"] = tmpName
            records.append(numericalize(row))

            if tmpName not in patternIDs:
                patternIDs.append(tmpName)
    return fieldNames, records, patternIDs


def addFieldNameQuantileMetadata(standardRecords, sampleRecords, patternID,
                                 readName, writeName):
    """For one pattern name, add its quantile data for one quantile type.

    For each mapped hit, appends quantile values calculated from a seqlet
    distribution of the considered score. These scores usually will consist of
    seq-match, contribution-match or contribution-magnitude scores.
    """
    standardValues = []
    for r in standardRecords:
        if r["_TMPNAME"] == patternID:
            standardValues.append(float(r[readName]))
    sampleValues = []
    for r in sampleRecords:
        if r["_TMPNAME"] == patternID:
            sampleValues.append(float(r[readName]))

    quantileMap = motifUtils.arrayQuantileMap(np.array(standardValues),
                                   np.array(sampleValues))
    readHead = 0
    for r in sampleRecords:
        if r["_TMPNAME"] == patternID:
            r[writeName] = quantileMap[readHead]
            readHead += 1


def addFieldQuantileData(standardRecords, sampleRecords, recordNames,
                         readName, writeName):
    """For one given field, populate the quantile data.

    For each pattern, appends quantile values calculated from a seqlet
    distribution of the considered score. These scores usually will consist of
    seq-match, contribution-match or contribution-magnitude scores.
    """
    for rn in recordNames:
        addFieldNameQuantileMetadata(standardRecords, sampleRecords, rn, readName,
                                     writeName)


def addAllMetadata(standardRecords, sampleRecords, recordNames, readNames, writeNames) -> None:
    """Add all of the quantile metadata.

    For each mapped hit, appends ALL quantile values calculated from the seqlet
    distribution of the considered score. These scores usually will consist of
    seq-match, contribution-match and contribution-magnitude scores.
    """
    for i, readName in enumerate(readNames):
        addFieldQuantileData(standardRecords, sampleRecords,
                             recordNames, readName, writeNames[i])


def writeTsv(records, fieldNames, fname) -> None:
    """Write a new set of mapped hits with the newly-appended quantile information."""
    with open(fname, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldNames, extrasaction="ignore",
                                delimiter="\t")
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def getParser() -> argparse.ArgumentParser:
    """Return the parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Given tsv files generated by motifSeqletCutoffs"
                    "and motifScan, add quantile information about each motif "
                    "hit for downstream analysis.")
    parser.add_argument("--seqlet-tsv", dest="seqletTsvFname", help="The name of the "
                        "seqlet tsv file generated by motifSeqletCutoffs.py "
                        "(or motifScan.py if run with pattern-cutoff-settings).",
                        required=True)
    parser.add_argument("--scan-tsv", dest="scanTsvFname", help="The name of the tsv "
                        "file generated by motifScan.py", required=True)
    parser.add_argument("--seqlet-out", dest="seqletOutFname", help="Instead of "
                        "overwriting seqlet-tsv, write the results to this file. If "
                        "omitted, edit seqlet-tsv in place.", required=False)
    parser.add_argument("--scan-out", dest="scanOutFname", help="Instead of overwriting "
                        "scan-tsv write them to this file. If omitted, edit scan-tsv in "
                        "place.",
                        required=False)
    parser.add_argument("--verbose", help="Include more debugging information.",
                        action="store_true")
    return parser


def main() -> None:
    """Add quantile information."""
    args = getParser().parse_args()
    logUtils.setBooleanVerbosity(args.verbose)
    seqletInFname = args.seqletTsvFname
    scanInFname = args.scanTsvFname
    seqletOutFname = args.seqletOutFname
    if seqletOutFname is None:
        seqletOutFname = seqletInFname
    scanOutFname = args.scanOutFname
    if scanOutFname is None:
        scanOutFname = scanInFname

    # Now read in the data.
    logUtils.info("Reading in seqlet tsvs")
    seqletFields, seqletRecords, seqletPatternNames = readTsv(seqletInFname)
    scanFields, scanRecords, scanPatternNames = readTsv(scanInFname)
    baseNames = ["contrib_magnitude", "seq_match", "contrib_match"]
    quantileNames = [x + "_quantile" for x in baseNames]
    # Check for existing quantile information.
    for qn in quantileNames:
        if qn in scanFields:
            # Weird, we already have a column for this data. Issue a warning.
            logUtils.warning(f"The tsv {scanInFname} seems to already have quantile columns.")
            break
    logUtils.info("Finished reading.")
    readNames = ["contrib_magnitude", "seq_match", "contrib_match"]
    writeNames = [x + "_quantile" for x in readNames]
    logUtils.info("Annotating seqlet tsvs")
    addAllMetadata(seqletRecords, seqletRecords, seqletPatternNames,
                   readNames, writeNames)
    logUtils.info("Annotating scanned hit tsvs")
    addAllMetadata(seqletRecords, scanRecords, scanPatternNames, readNames, writeNames)

    # Now we just have to write the records out.
    logUtils.info("Writing annotated seqlet tsv")
    writeTsv(seqletRecords, seqletFields + writeNames, seqletOutFname)
    logUtils.info("Writing annotated scan tsv")
    writeTsv(scanRecords, scanFields + writeNames, scanOutFname)


if __name__ == "__main__":
    main()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
