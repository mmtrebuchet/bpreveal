#!/usr/bin/env python3
import argparse
import pybedtools
import pysam
from bpreveal import logUtils
from bpreveal import bedUtils


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
    ap.add_argument("--all-chroms", help="Instead of listing the chromosomes "
                    "to use, use all of the chromosomes in the genome. "
                    "Cannot be used with --allow-chrom.",
                    action='store_true', dest='allChroms')
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
        logUtils.setVerbosity("DEBUG")
    else:
        logUtils.setVerbosity("WARNING")
    forbidRegions = []
    with pysam.FastaFile(args.genome) as genome:
        if args.allChroms:
            chroms = genome.references
        else:
            chroms = args.allowChrom
        for c in chroms:
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
        for remBl in args.blacklist:
            blacklist = blacklist.cat(pybedtools.BedTool(remBl))
    logUtils.info("Blacklist built.")
    with pysam.FastaFile(args.genome) as genome:
        whitelist = bedUtils.makeWhitelistSegments(genome, blacklist)
    logUtils.info("Whitelist built.")
    whitelist = pybedtools.BedTool([x for x in whitelist if x.chrom in chroms])
    whitelist = pybedtools.BedTool(list(whitelist))
    tiles = bedUtils.tileSegments(args.inputLength, args.outputLength, whitelist, args.spacing)
    tiles.saveas(args.outputBed)


if __name__ == "__main__":
    main()
