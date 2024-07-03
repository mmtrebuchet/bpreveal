#!/usr/bin/env python3
"""A simple utility to revcomp-augment data."""
import argparse
import pybedtools
import numpy as np
import pysam
import pyBigWig
from bpreveal import logUtils
from bpreveal import utils, bedUtils
logUtils.setVerbosity("INFO")


def getParser() -> argparse.ArgumentParser:
    """Build the parser (but don't parse_args())."""
    ap = argparse.ArgumentParser(description="A utility for generating "
        "reverse-complement test data.")
    ap.add_argument("--bed", help="The bed file containing regions to revcomp.")
    ap.add_argument("--input-length", help="The input length of your model.",
                    dest="inputLength", type=int)
    ap.add_argument("--input-bigwig", help="The input bigwig for reverse complement",
                    dest="inputBigwigFname")
    ap.add_argument("--output-bigwig", help="The output bigwig for reverse complement.",
                    dest="outputBigwigFname")
    ap.add_argument("--revcomp", help="Should the sequences be reverse-complemented?",
                    action="store_true")
    ap.add_argument("--genome", help="The genome fasta file.")
    ap.add_argument("--output-fasta", help="The name of the fasta-format file that should"
                    "be written.", dest="outputFastaFname")
    ap.add_argument("--output-bed", help="For regions that survived resizing, save them here.",
                    dest="outputBedFname")
    return ap


def main() -> None:
    """Generate the augmented data."""
    args = getParser().parse_args()

    regions = list(pybedtools.BedTool(args.bed))

    if args.outputBigwigFname is not None:
        # We've been asked to generate the revcomp bigwig.
        inBw = pyBigWig.open(args.inputBigwigFname, "r")
        chromSizes = utils.loadChromSizes(bw=inBw)
        logUtils.debug(str(chromSizes))
        inBwDats = {}
        for chromName in chromSizes.keys():
            vals = np.nan_to_num(inBw.values(chromName, 0, inBw.chroms(chromName)))
            inBwDats[chromName] = vals
        logUtils.debug("Loaded old data.")
        # Do we want to revcomp? Probably, since otherwise
        # this is a very slow way to copy a bigwig!
        if args.revcomp:
            for r in regions:
                origVals = inBwDats[r.chrom][r.start:r.end]
                flipVals = origVals[::-1]
                inBwDats[r.chrom][r.start:r.end] = flipVals
        logUtils.debug("Revcomped.")
        utils.writeBigwig(args.outputBigwigFname, chromDict=inBwDats)
        logUtils.debug("Saved.")

    if args.outputFastaFname is not None or args.outputBedFname is not None:
        resizedRegions = []
        passRegions = []
        genome = pysam.FastaFile(args.genome)
        for r in regions:
            if newRegion := bedUtils.resize(r, mode="center", width=args.inputLength,
                                           genome=genome):
                resizedRegions.append(newRegion)
                passRegions.append(r)
        if len(resizedRegions) != len(regions):
            # We lost some regions. Make sure we're saving a bed!
            assert args.outputBedFname is not None, \
                "Regions were removed, you have to save a bed."
        if args.outputBedFname is not None:
            passBt = pybedtools.BedTool(passRegions)
            passBt.saveas(args.outputBedFname)
        if args.outputFastaFname is not None:
            with open(args.outputFastaFname, "w") as fp:
                for r in resizedRegions:
                    fp.write(f">{r.chrom},{r.start},{r.end}\n")
                    seq = genome.fetch(r.chrom, r.start, r.end)
                    if args.revcomp:
                        oheSeq = utils.oneHotEncode(seq)
                        flipSeq = np.flip(oheSeq)
                        seq = utils.oneHotDecode(flipSeq)
                    fp.write(seq)
                    fp.write("\n")


if __name__ == "__main__":
    main()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
