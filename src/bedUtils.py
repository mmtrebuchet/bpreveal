"""Some utilities for dealing with bed files."""
import pybedtools
import pysam
import logging
import numpy as np
from typing import Literal
from bpreveal.utils import wrapTqdm


def makeWhitelistSegments(genome: pysam.FastaFile,
                          blacklist: pybedtools.BedTool | None = None) -> pybedtools.BedTool:
    """Get a list of windows where it is safe to draw inputs for your model.

    :param genome: A FastaFile (pysam object, not a string filename!).
    :param blacklist: (Optional) A BedTool that gives additional regions that should
        be excluded.
    :return: A BedTool that contains the whitelisted regions.

    Given a genome file, go over each chromosome and see where the Ns are.
    Create a BedTool that contains all regions that don't contain N.
    For example, if your genome were
    ``ATATATATnnnnnnnATATATATATATnnn``,
    then this would return a BedTool corresponding to the regions containing
    As and Ts.

    ``blacklist``, if provided, is a bed file of regions that should be treated as though
    they contained N nucleotides.
    """
    segments = []

    logging.debug("Building segments.")
    blacklistsByChrom = dict()
    if blacklist is not None:
        # We're going to iterate over blacklist several times,
        # so save it in case it's a streaming bedtool.
        blacklist.saveas()
        for blackInterval in blacklist:
            if blackInterval.chrom not in blacklistsByChrom:
                blacklistsByChrom[blackInterval.chrom] = []
            blacklistsByChrom[blackInterval.chrom].append(blackInterval)

    for chromName in wrapTqdm(sorted(genome.references), "INFO"):
        chromSeq = genome.fetch(chromName, 0, genome.get_reference_length(chromName))
        if chromName in blacklistsByChrom:
            seqList = list(chromSeq)
            for blackInterval in blacklistsByChrom[chromName]:
                for base in range(blackInterval.start, blackInterval.end - 1):
                    try:
                        seqList[base] = 'N'
                    except IndexError:
                        logging.warning("Ran off the end of the chromosome. Interval: {0:s}"
                                        .format(str(blackInterval)))
                        break
            chromSeq = ''.join(seqList)
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
            # We finished the chromosome without hitting Ns. Certainly possible!
            segments.append(pybedtools.Interval(chromName, segmentStart, len(chromSeq)))
    return segments


def tileSegments(inputLength: int, outputLength: int,
                 segments: pybedtools.BedTool,
                 spacing: int) -> pybedtools.BedTool:
    """Tile the given segments with intervals.

    :param inputLength: The input-length of your model.
    :param outputLength: The output-length of your model, and also the length of the
        intervals in the returned ``BedTool``.
    :param segments: The regions of the genome that you'd like tiled.
    :param spacing: The distance *between* the windows.
    :return: A BedTool containing Intervals of length ``outputLength``.


    ``segments`` will often come from :func:`makeWhitelistSegments`.

    ``spacing`` is the distance between the *end* of one region and the *start* of the next.
    So to tile the whole genome, set ``spacing=0``. ``spacing`` may be negative, in which
    case the tiled regions will overlap.

    When this algorithm reaches the end of a segment, it will try to place an additional
    region if it can. For example, if your window is 30 bp long, with outputLength 6,
    inputLength 10, and spacing 5 you'd get::

        012345678901234567890123456789
        --xxxxxx-----xxxxxx---xxxxxx--

    The 2 bp padding on each end comes from the fact that
    ``(inputLength - outputLength) / 2 == 2``
    Note how the last segment is not 5 bp away from the second-to-last.

    """
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
    logging.debug("Filtered segments. {0:d} survive.".format(shrunkSegments.count()))

    # Phase 3. Generate tiling regions.
    logging.debug("Creating regions.")
    regions = []
    for s in wrapTqdm(shrunkSegments, "INFO"):
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


def createTilingRegions(inputLength: int, outputLength: int,
                        genome: pysam.FastaFile,
                        spacing: int) -> pybedtools.BedTool:
    """Create a list of regions that tile a genome.

    :param inputLength: The input-length of your model.
    :param outputLength: The output-length of your model.
    :param genome: A FastaFile (the pysam object, not a string!)
    :param spacing: The space you'd like *between* returned intervals.
    :return: A BedTool containing regions that tile the genome.

    The returned BedTool will contain regions that are outputLength wide,
    and all regions will be far enough away from any N nucleotides that
    there will be no Ns in the input to your model.
    spacing specifies the amount of space *between* the regions. A spacing of 0
    means that the regions should join end-to-end, while a spacing of -500 would indicate
    regions that overlap by 500 bp.
    See :func:`tileSegments` for details on how the regions are placed.

    """
    # Segments are regions of the genome that contain no N nucleotides.
    # These will be split into regions in the next phase.
    segments = makeWhitelistSegments(genome)
    return tileSegments(inputLength, outputLength, segments, spacing)


def resize(interval: pybedtools.Interval, mode: str, width: int,
           genome: pysam.FastaFile) -> pybedtools.Interval | Literal[False]:
    """Resize a given interval to a new size.

    :param interval: A pyBedTools Interval object.
    :param mode: One of ``"none"``, ``"center"``, or ``"start"``.
    :param width: How long the returned Interval will be.
    :param genome: A FastaFile (the pysam object, not a string)
    :return: A newly-allocated Interval of the correct size, or ``False``
        if resizing would run off the edge of a chromosome.

    Given an interval (a PyBedTools Interval object),
    return a new Interval that is at the same coordinate.

    mode is one of:

    * "none", meaning that no resizing is done. In that case, this function will
      check that the interval obeys stop-start == width. If an interval
      does not have the correct width, an assertion will fail.
    * "center", in which case the interval is resized around its center.
    * "start", in which case the start coordinate is preserved.

    The returned interval will obey x.end - x.start == width.
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
    """Get the total counts from all bigwigs at a given Interval.

    :param interval: A pyBedTools Interval.
    :param bigwigs: A list of opened pyBigWig objects (not strings!).
    :return: A single number giving the total reads from all bigwigs at
        the given interval.

    NaN entries in the bigwigs are treated as zero.
    """
    total = 0
    for bw in bigwigs:
        vals = np.nan_to_num(bw.values(interval.chrom, interval.start, interval.end))
        total += np.sum(vals)
    return total


def sequenceChecker(interval: pybedtools.Interval, genome: pysam.FastaFile) -> bool:
    """For the given interval, does it only contain A, C, G, and T?

    :param interval: The interval to check.
    :param genome: A FastaFile (pysam object, not a string!).
    :return: ``True`` if the sequence matches ``"^[ACGTacgt]*$"``,
        ``False`` otherwise.
    """
    seq = genome.fetch(interval.chrom, interval.start, interval.end)
    if (len(seq.upper().lstrip('ACGT')) != 0):
        # There were letters that aren't regular bases. (probably Ns)
        return False
    return True


def lineToInterval(line: str) -> pybedtools.Interval | Literal[False]:
    """Take a line from a bed file and create a PyBedTools Interval object.

    :param line: The line from the bed file
    :return: A newly-allocated pyBedTools Interval object, or ``False`` if
        the line is not a valid bed line.
    """
    if len(line.strip()) == 0 or line[0] == '#':
        return False
    initInterval = pybedtools.cbedtools.create_interval_from_list(line.split())
    return initInterval
