"""Some utilities for dealing with bed files"""
import pybedtools
import pysam
import pyBigWig
import logging
import numpy as np
from typing import Literal


def createTilingRegions(inputLength: int, outputLength: int,
                        genome: pysam.FastaFile,
                        spacing: int) -> pybedtools.BedTool:
    """
    Create a list of regions that tile a genome.

    inputLength and outputLength have the same meaning as for models.
    The returned BedTool will contain regions that are outputLength wide,
    and all regions will be far enough away from any N nucleotides that
    there will be no Ns in the input to your model.
    spacing specifies the amount of space BETWEEN the regions. A spacing of 0
    means that the regions should join end-to-end, while a spacing of -500 would indicate
    regions that overlap by 500 bp.

    Returns a BedTool.
    """
    # Segments are regions of the genome that contain no N nucleotides.
    # These will be split into regions in the next phase.
    segments = []
    logging.debug("Building segments.")
    for chromName in sorted(genome.references):
        chromSeq = genome.fetch(chromName, 0, genome.get_reference_length(chromName))
        segmentStart = 0
        inSegment = False
        for i, c in enumerate(chromSeq):
            if c not in 'ACGTacgt':
                # We have hit a segment boundary.
                if inSegment:
                    # We should commit the current segment.
                    segments.append(pybedtools.Interval(chromName, segmentStart, i))
                inSegment = False
            elif not inSegment:
                # Start up a new segment.
                segmentStart = i
                inSegment = True

        if inSegment:
            # We finished the chromosome with hitting Ns.
            segments.append(pybedtools.Interval(chromName, segmentStart, len(chromSeq)))
    # Phase two: trim the segments.
    logging.debug("Beginning to trim segments. {0:d} segments alive.".format(len(segments)))
    padding = (inputLength - outputLength) // 2
    logging.debug("Calculated padding of {0:d}".format(padding))

    def shrinkSegment(s: pybedtools.Interval):
        newEnd = s.end - padding
        newStart = s.start + padding
        if newEnd - newStart < outputLength:
            return False
        return pybedtools.Interval(s.chrom, newStart, newEnd)

    shrunkSegments = pybedtools.BedTool(segments).each(shrinkSegment).saveas()
    shrunkSegments.saveas("/dev/shm/segments.bed")
    logging.debug("Filtered segments. {0:d} survive.".format(shrunkSegments.count()))

    # Phase 3. Generate tiling regions.
    logging.debug("Creating regions.")
    regions = []
    for s in shrunkSegments:
        startPos = s.start
        endPos = startPos + outputLength
        while endPos < s.end:
            curRegion = pybedtools.Interval(s.chrom, startPos, endPos)
            regions.append(curRegion)
            startPos += spacing + outputLength
            endPos = startPos + outputLength
        if startPos < s.end:
            # We want another region inside this segment.
            endPos = s.end
            startPos = endPos - outputLength
            regions.append(pybedtools.Interval(s.chrom, startPos, endPos))
    logging.debug("Regions created, {0:d} across genome.".format(len(regions)))

    return pybedtools.BedTool(regions)


def resize(interval: pybedtools.Interval, mode: str, width: int,
           genome: pysam.FastaFile) -> pybedtools.Interval | Literal[False]:
    """Given an interval (a PyBedTools Interval object),
    return a new Interval that is at the same coordinate.
    (see mode for the meaning of "same").
    Arguments:
    Interval is a PyBedTools Interval object with start and end information.
    mode is one of:
        "none", meaning that no resizing is done. In that case, this function will
            check that the interval obeys stop-start == width. If an interval
            does not have the correct width, an assertion will fail.
        "center", in which case the interval is resized around its center.
        "start", in which case the start coordinate is preserved.
    width is an integer. The returned interval will obey x.end - x.start == width.
    genome is an opened pysam genome fasta file. This is used to check if an interval
        has fallen off the edge of a chromosome. If this is the case, this function will
        return False. Check for this!
    Returns:
        A PyBedTools Interval object, newly allocated.
        It will preserve the chromosome, name, score, and strand
        information, but not other bed fields.
    """
    start = interval.start
    end = interval.end
    match mode:
        case "none":
            if (end - start != width):
                assert False, \
                       "An input region is not the expected width: {0:s}".format(str(interval))
        case "center":
            center = (end + start) // 2
            start = center - width // 2
            end = start + width
        case "start":
            start = start - width // 2
            end = start + width
        case _:
            assert False, "Unsupported resize mode: {0:s}".format(mode)
    if (start <= 0 or end >= genome.get_reference_length(interval.chrom)):
        return False  # We're off the edge of the chromosome.
    return pybedtools.Interval(interval.chrom, start, end, name=interval.name,
                               score=interval.score, strand=interval.strand)


def getCounts(interval: pybedtools.Interval, bigwigs: list) -> float:
    """For the given PyBedTools interval and a list of open bigwig files
    (NOT file names, actual file objects), determine
    the SUM of counts for that interval across all given bigwigs.
    Returns:
    A single floating-point value representing the total of each bigwig in the given region.
    NaN entries in the bigwigs are treated as zero.
    """
    total = 0
    for bw in bigwigs:
        vals = np.nan_to_num(bw.values(interval.chrom, interval.start, interval.end))
        total += np.sum(vals)
    return total


def sequenceChecker(interval: pybedtools.Interval, genome: pysam.FastaFile) -> bool:
    """For the given interval, does it only contain A, C, G, and T?
    If there are other bases, like N, returns False.
    Returns:
        True if the sequence matches "^[ACGTacgt]*$", False otherwise.
    """
    seq = genome.fetch(interval.chrom, interval.start, interval.end)
    if (len(seq.upper().lstrip('ACGT')) != 0):
        # There were letters that aren't regular bases. (probably Ns)
        return False
    return True


def lineToInterval(line: str) -> pybedtools.Interval | Literal[False]:
    """Simply takes a text line from a bed file and creates a PyBedTools Interval object.
    If the line is not a data line, return False."""
    if len(line.strip()) == 0 or line[0] == '#':
        return False
    initInterval = pybedtools.cbedtools.create_interval_from_list(line.split())
    return initInterval
