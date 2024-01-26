#!/usr/bin/env python3
import argparse
from bpreveal import bedUtils
import pybedtools
import pysam
import logging
from bpreveal import utils
utils.setVerbosity("DEBUG")


def getParser():
    ap = argparse.ArgumentParser(
        description="Little tool to generate regions that tile the genome, "
                    "excluding a set of regions that you specify.")
    ap.add_argument("--genome", help="The fasta-format genome file to tile.")
    ap.add_argument("--allow-chrom",
        help="A chromosome to allow in your tiled set. "
             "May be given multiple times.",
        action='append',
        dest='allowChrom')
    ap.add_argument("--output-length", help="The width of the output regions.",
                    type=int, dest='outputLength')
    ap.add_argument("--input-length", help="The width of the model input.",
                    type=int, dest='inputLength')

    ap.add_argument("--chrom-edge-boundary",
        help="How far from the chromosome edges should the tiled regions start?",
        type=int,
        dest='chromEdgeBoundary')
    ap.add_argument("--spacing", help="The space between regions.", type=int)
    ap.add_argument("--output-bed", help="The bed file that will be written.", dest='outputBed')
    ap.add_argument("--blacklist",
        help="A bed file of regions to exclude. May be given multiple times.",
        action='append')
    ap.add_argument("--verbose", help="Display progress messages", action='store_true')
    return ap


def main():
    args = getParser().parse_args()
    if args.verbose:
        utils.setVerbosity("DEBUG")
    else:
        utils.setVerbosity("WARNING")
    forbidRegions = []
    with pysam.FastaFile(args.genome) as genome:
        for c in args.allowChrom:
            chromLen = genome.get_reference_length(c)
            if chromLen < 2 * args.chromEdgeBoundary:
                forbidRegions.append(
                    pybedtools.Interval(c, 0, chromLen))
                continue
            forbidRegions.append(pybedtools.Interval(c, 0, args.chromEdgeBoundary))
            forbidRegions.append(
                pybedtools.Interval(c, chromLen - args.chromEdgeBoundary, chromLen))
    blacklist = pybedtools.BedTool(forbidRegions)
    if args.blacklist is not None:
        blacklist = pybedtools.BedTool(args.blacklist[0])
        for remBl in args.blacklist[1:]:
            blacklist = blacklist.cat(pybedtools.BedTool(remBl))
    logging.info("Blacklist built.")
    with pysam.FastaFile(args.genome) as genome:
        whitelist = bedUtils.makeWhitelistSegments(genome, blacklist)
    logging.info("Whitelist built.")
    whitelist = pybedtools.BedTool([x for x in whitelist if x.chrom in args.allowChrom])
    whitelist = pybedtools.BedTool(list(whitelist))
    tiles = bedUtils.tileSegments(args.inputLength, args.outputLength, whitelist, args.spacing)
    tiles.saveas(args.outputBed)


if __name__ == "__main__":
    main()
