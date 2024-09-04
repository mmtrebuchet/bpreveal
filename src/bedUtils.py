"""Some utilities for dealing with bed files."""
from typing import Literal, Any
import multiprocessing
from collections import deque
import os
import pybedtools
import pysam
import numpy as np
import pyBigWig
from bpreveal import logUtils
from bpreveal.logUtils import wrapTqdm
from bpreveal.internal import constants
from bpreveal.internal.crashQueue import CrashQueue


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

    logUtils.debug("Building segments.")
    blacklistsByChrom = {}
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
        seqVector = np.fromstring(chromSeq, np.int8)  # type: ignore
        if chromName in blacklistsByChrom:
            for blackInterval in blacklistsByChrom[chromName]:
                if blackInterval.start >= seqVector.shape[0]:
                    continue
                endPt = min(seqVector.shape[0], blackInterval.end)
                seqVector[blackInterval.start:endPt] = ord("N")
        segments.extend(_findNonN(seqVector, chromName))
    return pybedtools.BedTool(segments)


def _findNonN(inSeq: np.ndarray, chromName: str) -> list[pybedtools.Interval]:
    """Return a list of Intervals consisting of all regions of the sequence that are not N.

    :param inSeq: an array of character values - not a one-hot encoded sequence::

        inSeq[i] = ord(dnaStr[i])

    :param chromName: Just the name of the chromosome, used to populate the chrom
        field in the returned Interval objects.

    :return: A list of Intervals where the sequence is NOT ``n`` or ``N``.
    """
    segments = []
    # All bases that are not N.
    isValid = np.logical_not(np.logical_or(inSeq == ord("N"), inSeq == ord("n")))
    # ends is 1 if this base is the end of a valid region, 0 otherwise.
    ends = np.empty(isValid.shape, dtype=np.bool_)
    # starts is 1 if this base is the beginning of a valid region, 0 otherwise.
    starts = np.empty(isValid.shape, dtype=np.bool_)
    starts[0] = isValid[0]
    ends[-1] = isValid[-1]

    ends[:-1] = np.logical_and(isValid[:-1], np.logical_not(isValid[1:]))
    starts[1:] = np.logical_and(isValid[1:], np.logical_not(isValid[:-1]))
    # the Poses are the actual indices where the start and end vectors are nonzero.
    endPoses = np.flatnonzero(ends)  # type: ignore
    startPoses = np.flatnonzero(starts)  # type: ignore
    for startPos, stopPos in zip(startPoses, endPoses):
        segments.append(pybedtools.Interval(chromName, startPos, stopPos + 1))
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
    logUtils.debug(f"Beginning to trim segments. {len(segments)} segments alive.")
    padding = (inputLength - outputLength) // 2
    logUtils.debug(f"Calculated {padding=}")

    def shrinkSegment(s: pybedtools.Interval) -> pybedtools.Interval:
        newEnd = s.end - padding
        newStart = s.start + padding
        if newEnd - newStart < outputLength:
            return False
        return pybedtools.Interval(s.chrom, newStart, newEnd)

    shrunkSegments = pybedtools.BedTool(segments).each(shrinkSegment).saveas()
    logUtils.debug(f"Filtered segments. {shrunkSegments.count()} survive.")

    # Phase 3. Generate tiling regions.
    logUtils.debug("Creating regions.")
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
    logUtils.debug(f"Regions created, {len(regions)} across genome.")

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

    * ``"none"``, meaning that no resizing is done. In that case, this function will
      check that the interval obeys stop-start == width. If an interval
      does not have the correct width, an assertion will fail.
    * ``"center"``, in which case the interval is resized around its center.
    * ``"start"``, in which case the start coordinate is preserved.

    The returned interval will obey ``x.end - x.start == width``.
    It will preserve the chromosome, name, score, and strand
    information, but not other bed fields.
    """
    start = interval.start
    end = interval.end
    match mode:
        case "none":
            if end - start != width:
                raise ValueError(f"An input region is not the expected width: {interval}")
        case "center":
            center = (end + start) // 2
            start = center - width // 2
            end = start + width
        case "start":
            start = start - width // 2
            end = start + width
        case _:
            raise ValueError(f"Unsupported resize mode: {mode}")
    if start <= 0 or end >= genome.get_reference_length(interval.chrom):
        return False  # We're off the edge of the chromosome.
    return pybedtools.Interval(interval.chrom, start, end, name=interval.name,
                               score=interval.score, strand=interval.strand)


def _metapeakThread(bwFname: str,
                    inQueue: CrashQueue,
                    outQueue: CrashQueue) -> None:
    bigwigFp = pyBigWig.open(bwFname)
    totalProfile = None
    numQueries = 0
    while (query := inQueue.get()) is not None:
        numQueries += 1
        chrom, start, end, strand = query
        profile = np.nan_to_num(bigwigFp.values(chrom, start, end))
        if strand == "-":
            profile = np.flip(profile)
        if totalProfile is None:
            totalProfile = profile
        else:
            totalProfile = totalProfile + profile
        outQueue.put((totalProfile, numQueries))
    bigwigFp.close()


def metapeak(intervals: pybedtools.BedTool,
             bigwigFname: str, numThreads: int | None = None) -> constants.PRED_AR_T:
    """Go over the given intervals and build a metapeak.

    :param intervals: A pyBigWig file containing the regions to use.
        This can also be a list of Interval objects. Each interval
        must be of the same size.
    :param bigwigFname: The name of the bigwig file to read in.
    :param numThreads: If provided, the number of parallel workers to use.
        If not specified, use all of the CPUs on the machine.
    :return: A numpy array of shape (interval-length,) containing the average
        profile over all of the intervals.

    This produces a stranded metapeak. If an interval is on the negative
    strand, then the values extracted from the bigwig will be reversed before
    being added to the metapeak.

    The parallel implementation is memory-efficient and can easily scale to metapeaks
    with millions of underlying regions. However, this means that interrupting the
    calculation can leave the program in a really ugly state. I recommend checking
    your inputs before you call this.

    NaN entries in the bigwig are treated as zero.
    """
    # I don't usually do error checking, but getting a crash in this function
    # could leave the interpreter in a tizzy, and it will likely be used
    # interactively from jupyter, so a weird interpreter state could persist.
    intervalList = list(intervals)  # Copy the BedTool in case it's a streaming one.
    width = intervalList[0].end - intervalList[0].start
    for i in intervalList:
        assert i.end - i.start == width, \
            f"Detected an interval with wrong width: {str(i)}, expected {width}"
    # Make sure we can open the bigwig (then close it).
    pyBigWig.open(bigwigFname).close()

    # Okay, sanity checks passed. Time to actually run the calculation.
    if numThreads is None:
        numThreads = len(os.sched_getaffinity(0))
    pids = []
    inQueue = CrashQueue()
    outQueue = CrashQueue()
    for i in range(numThreads):
        pids.append(multiprocessing.Process(
            target=_metapeakThread,
            args=(bigwigFname, inQueue, outQueue),
            daemon=True))
        pids[i].start()
    for interval in intervalList:
        inQueue.put((interval.chrom, interval.start, interval.end, interval.strand))
    for _ in range(numThreads):
        inQueue.put(None)  # We're done, send the termination signal.
    rets = []
    for _ in range(numThreads):
        rets.append(outQueue.get())
    for i in range(numThreads):
        pids[i].join()
    totalProfile = rets[0][0]
    totalCounts = rets[0][1]
    for p, c in rets[1:]:
        totalProfile += p
        totalCounts += c
    totalProfile /= totalCounts
    return totalProfile


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
    if len(seq.upper().lstrip("ACGT")) != 0:
        # There were letters that aren't regular bases. (probably Ns)
        return False
    return True


def lineToInterval(line: str) -> pybedtools.Interval | Literal[False]:
    """Take a line from a bed file and create a PyBedTools Interval object.

    :param line: The line from the bed file
    :return: A newly-allocated pyBedTools Interval object, or ``False`` if
        the line is not a valid bed line.
    """
    if len(line.strip()) == 0 or line[0] == "#":
        return False
    initInterval = pybedtools.cbedtools.create_interval_from_list(line.split())
    return initInterval


class ParallelCounter:
    """A class that queues up :py:func:`~getCounts` jobs and runs them in parallel.

    This is used by the :py:mod:`prepareBed<bpreveal.prepareBed>` script.

    :param bigwigNames: The name of the bigwig files to read from
    :param numThreads: How many parallel workers should be used?
    """

    def __init__(self, bigwigNames: list[str], numThreads: int):
        self.bigwigNames = bigwigNames
        self.numThreads = numThreads
        self.inQueue = CrashQueue()
        self.outQueue = CrashQueue()
        self.inFlight = 0
        self.outDeque = deque()
        self.numInDeque = 0
        self.threads = [multiprocessing.Process(
            target=_counterThread,
            args=(bigwigNames, self.inQueue, self.outQueue))
            for _ in range(numThreads)]
        for t in self.threads:
            t.start()

    def addQuery(self, query: tuple[str, int, int], idx: Any) -> None:
        """Add a region (chrom, start, end) to the task list.

        :param query: A tuple of (chromosome, start, end) giving the region to look at.
        :param idx: An index that will be returned with the results.
        """
        self.inQueue.put(query + (idx,))
        self.inFlight += 1
        while not self.outQueue.empty():
            self.outDeque.appendleft(self.outQueue.get())
            self.numInDeque += 1
            self.inFlight -= 1

    def done(self) -> None:
        """Wrap up the show - close the child threads."""
        for _ in range(self.numThreads):
            self.inQueue.put(None)
        for t in self.threads:
            t.join()

    def getResult(self) -> tuple[float, int]:
        """Get the next result.

        :return: A tuple of (counts, idx), where counts is the total counts
            for the bigwigs and idx is the index of the region you gave to addQuery.

        Note that the results will NOT be given in order - you must look at the index
        values.
        """
        if self.inFlight and self.numInDeque == 0:
            self.outDeque.appendleft(self.outQueue.get())
            self.numInDeque += 1
            self.inFlight -= 1
        self.numInDeque -= 1
        return self.outDeque.pop()


def _counterThread(bigwigFnames: list[str], inQueue: CrashQueue,
                   outQueue: CrashQueue) -> None:
    """Thread to sum up regions of the bigwigs.

    :param bigwigFnames: A list of file names to open.
    :param inQueue: The input queue, where queries will come from.
    :param outQueue: Where the calculated counts should be put.
    :return: None, but does put results in outQueue.

    The runner, :py:class:`~ParallelCounter`, will inject regions
    in the format ``tuple[str, int, int, Any]``, which contains, in order,

    1. Chromosome (``str``), the chromosome that the region is on.
    2. Start (``int``), the start coordinate, 0-based, inclusive.
    3. End (``int``), the end coordinate, 0-based, exclusive.
    4. Index, which will be passed back with the result.

    The results put counts data into ``outQueue``, with format
    ``tuple[int, Any]``, containing:

    1. Total counts, a float.
    2. Index, which is straight from the input queue.

    The total counts for a region is specified by::

        def total(chrom, start, end):
            total = 0.0
            for bw in bigwigs:
                total += sum(abs(bw.values(chrom, start, end)))
            return total

    """
    bwFiles = [pyBigWig.open(fname) for fname in bigwigFnames]
    outDeque = deque()
    inDeque = 0
    while True:
        query = inQueue.get()
        match query:
            case (chrom, start, end, idx):
                r = getCounts(pybedtools.Interval(chrom, start, end), bwFiles)
                outDeque.appendleft((r, idx))
                inDeque += 1
                while outQueue.empty and inDeque > 0:
                    outQueue.put(outDeque.pop())
                    inDeque -= 1
            case None:
                break
    while inDeque:
        outQueue.put(outDeque[-1])
        inDeque -= 1
    for bwFp in bwFiles:
        bwFp.close()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
