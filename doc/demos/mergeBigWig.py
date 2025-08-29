#!/usr/bin/env python3
"""A little script to add multiple bigwig files together.

This is mostly to combine replicates.
"""
import argparse
import numpy as np
import pyBigWig
from bpreveal import utils


def addBws(bwFnames: list[str], outFname: str, chromSizes: str) -> None:
    """Take the given bigwig files and add their values. Save the result.

    :param bwFnames: The file names for the input bigwigs.
    :param outFname: The name of the bigwig file to write.
    """
    bws = [pyBigWig.open(x) for x in bwFnames]
    bwOut = pyBigWig.open(outFname, "w")
    bwChroms = utils.loadChromSizes(chromSizesFname=chromSizes)
    header = [(x, bwChroms[x]) for x in sorted(bwChroms.keys())]
    bwOut.addHeader(header)

    for chromName in sorted(bwChroms.keys()):
        datOut = np.zeros((bwChroms[chromName],), dtype=np.float32)
        for bw in bws:
            if chromName in bw.chroms():
                d = np.nan_to_num(bw.values(chromName, 0, bw.chroms()[chromName]))
                datOut[:d.shape[0]] += d
        bwOut.addEntries(chromName,
                          0,
                          values = [float(x) for x in datOut],
                          span=1,
                          step=1)

def main():
    ap = argparse.ArgumentParser(description="Add multiple bigwig files together.")
    ap.add_argument("inFiles", nargs='+', help="Input bigwig files to read.")
    ap.add_argument("--output", help="The output file to write.")
    ap.add_argument("--chrom-sizes", dest='chromSizes', help="A file giving the chromosome sizes for the genome.")
    args = ap.parse_args()
    addBws(args.inFiles, args.output, args.chromSizes)


if __name__ == "__main__":
    main()
