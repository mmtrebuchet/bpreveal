#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
from bpreveal.utils import oneHotDecode


def getParser():
    ap = argparse.ArgumentParser(description="Read in a modisco hdf5 file and write a "
                                 "fastq file containing the seqlets to stdout.")
    ap.add_argument("--modisco-h5", help="The modisco hdf5 file", dest="modiscoH5")
    ap.add_argument("--contrib-h5", help="The contribution score hdf5 file", dest="contribH5")
    ap.add_argument("--bed",
                    help="The name of the output bed file with coordinates from the contrib h5.")
    ap.add_argument("--fastq", help="The name of a fastq file to write with the sequences.")
    ap.add_argument("--width", help="The width of windows used by modisco. "
                    "(The -w argument to modisco motifs)",
                    type=int, default=0)
    return ap


def dumpGroup(subpattern, contribFp, name, bedFp, fastqFp, width):
    seqlets = np.array(subpattern["seqlets"]["sequence"])
    revcomps = np.array(subpattern["seqlets"]["is_revcomp"])
    startPoses = np.array(subpattern["seqlets"]["start"])
    endPoses = np.array(subpattern["seqlets"]["end"])
    exampleIdxes = np.array(subpattern["seqlets"]["example_idx"])
    for i in range(seqlets.shape[0]):
        seq = seqlets[i]
        if revcomps[i]:
            seq = np.flip(seq)
        strandChr = "-" if revcomps[i] else "+"
        curSequence = oneHotDecode(seq)
        startPos = startPoses[i]
        endPos = endPoses[i]
        exampleIdx = exampleIdxes[i]
        seqletName = f"{name}_{i}_{strandChr}_{startPos}_{exampleIdx}"
        if fastqFp is not None:
            fastqFp.write(f"@{seqletName}\n")
            fastqFp.write(curSequence + "\n")
            fastqFp.write(f"+{seqletName}\n")
            fastqFp.write("z" * len(curSequence) + "\n")
        if contribFp is not None and bedFp is not None:
            assert width > 0, "Need to specify a width if you want a bed output."
            chrom = contribFp["coords_chrom"][exampleIdx]
            chromName = contribFp["chrom_names"].asstr()[chrom]
            startPt = contribFp["coords_start"][exampleIdx]
            endPt = contribFp["coords_end"][exampleIdx]
            center = (startPt + endPt) // 2
            windowStart = center - width // 2
            regionStart = windowStart + min(startPos, endPos)
            regionEnd = windowStart + max(startPos, endPos)
            bedFp.write(f"{chromName}\t{regionStart}\t{regionEnd}\t{seqletName}\t0\t{strandChr}\n")


def main():
    args = getParser().parse_args()
    with h5py.File(args.modiscoH5, "r") as fp:
        outFastq = None
        outBed = None
        contribFp = None
        if args.bed is not None:
            outBed = open(args.bed, "w")
        if args.fastq is not None:
            outFastq = open(args.fastq, "w")
        if args.contribH5 is not None:
            contribFp = h5py.File(args.contribH5, "r")
        for metaclusterTitle, mc in fp.items():
            mcName = metaclusterTitle.split("_")[0]
            for patternTitle, pat in mc.items():
                patName = patternTitle.split("_")[1]
                for subpatTitle in pat.keys():
                    if subpatTitle.startswith("subpattern"):
                        sp = pat[subpatTitle]
                        spName = subpatTitle.split("_")[1]
                        dumpGroup(sp, contribFp, f"{mcName}_{patName}_{spName}",
                                  outBed, outFastq, args.width)


if __name__ == "__main__":
    main()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
