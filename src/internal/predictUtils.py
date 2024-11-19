#!/usr/bin/env python3
"""Some functions and classes for streaming predictions."""
import numpy as np
import h5py
import pybedtools
import pysam
from bpreveal import logUtils
from bpreveal.logUtils import wrapTqdm
from bpreveal.internal.constants import LOGIT_T, LOGCOUNT_T
import bpreveal.internal.files


class FastaReader:
    """Streams a fasta file from disk lazily.

    :param fastaFname: The name of the fasta file to load.
    """

    curSequence = ""
    """The current sequence in this file. Updated by :py:meth:`~pop`."""
    curLabel = ""
    """The current description line in this file. Updated by :py:meth:`~pop`."""
    numPredictions = 0
    _nextLabel = ""

    def __init__(self, fastaFname: str):
        """Scan the file to count the total lines, then load the first sequence."""
        # First, scan over the file and count how many sequences are in it.
        logUtils.info("Counting number of samples.")
        with open(fastaFname, "r") as fp:
            for inLine in wrapTqdm(fp):
                line = inLine.strip()  # Get rid of newlines.
                if len(line) == 0:
                    continue  # There is a blank line. Ignore it.
                if line[0] == ">":
                    self.numPredictions += 1
            fp.close()
        self._idx = -1  # We're about to pop to get index zero.
        logUtils.info(f"Found {self.numPredictions} entries in input fasta")
        # Note that we close the file after that with block, so this re-opens it
        # at position zero.
        self._fp = open(fastaFname, "r")  # pylint: disable=consider-using-with
        # We know a fasta starts with >, so read in the first label.
        self._nextLabel = self._fp.readline().strip()[1:]
        self.pop()  # Read in the first sequence.

    def pop(self) -> None:
        """Pop the current sequence off the queue. Updates curSequence and curLabel."""
        # We know we're at the start of the sequence section of a fasta.
        self._idx += 1
        if self._idx >= self.numPredictions:
            logUtils.debug("Reached end of fasta generator.")
            return
        self.curLabel = self._nextLabel
        self.curSequence = ""
        inSequence = True
        curLine = self._fp.readline()
        while inSequence and len(curLine) > 1:
            if curLine[0] != ">":
                self.curSequence = self.curSequence + curLine.strip()
            else:
                self._nextLabel = curLine[1:].strip()
                inSequence = False
                break
            curLine = self._fp.readline()


class BedReader:
    """Streams a bed file from disk and loads sequence information lazily.

    :param bedFname: The name of the fasta file to load.
    :param genomeFname: The name of the fasta-format genome file.
    :param padding: The amount by which each region should be expanded before
        fetching the sequence. This will be (inputLength - outputLength) // 2
        for most cases.
    """

    numPredictions: int = 0
    """The total number of regions in the bed file."""
    curSequence: str
    """The genomic sequence under the current region.

    This will update when you call pop().
    """
    curChrom: str
    curStart: int
    curEnd: int

    curLabel = ""
    """Just for compatibility with the FastaReader, this will always be an empty string."""

    def __init__(self, bedFname: str, genomeFname: str, padding: int):
        """Scan the bed file to count the total lines."""
        logUtils.debug("Counting number of samples.")
        self._bt = list(pybedtools.BedTool(bedFname))
        self.numPredictions = len(self._bt)
        logUtils.info(f"Found {self.numPredictions} entries in input bed file")
        self._idx = 0
        self.genome = pysam.FastaFile(genomeFname)
        self.padding = padding
        self._fetch()

    def _fetch(self) -> None:
        r = self._bt[self._idx]
        s = self.genome.fetch(r.chrom, r.start - self.padding, r.end + self.padding)
        self.curSequence = s.upper()
        self.curChrom = r.chrom
        self.curStart = r.start
        self.curEnd = r.end

    def pop(self) -> None:
        """Pop the current sequence off the queue."""
        self._idx += 1
        if self._idx >= self.numPredictions:
            logUtils.debug("Reached end of bed generator.")
        else:
            self._fetch()


class H5Writer:
    """Batches up predictions and saves them in chunks.

    :param fname: The name of the hdf5 file to save.
    :param numHeads: The total number of heads for this model.
    :param numPredictions: How many total predictions will be made?

    """

    def __init__(self, fname: str, numHeads: int, numPredictions: int,
                 bedFname: str | None = None, genomeFname: str | None = None,
                 config: str | None = None):
        """Load everything that can be loaded before the subprocess launches."""
        self._fp = h5py.File(fname, "w")
        bpreveal.internal.files.addH5Metadata(self._fp, config=str(config))
        self.numHeads = numHeads
        self.numPredictions = numPredictions
        self.writeHead = 0
        self.batchWriteHead = 0
        self.writeChunkSize = 100
        if bedFname is not None:
            logUtils.info("Adding coordinate information.")
            assert genomeFname is not None, "Must supply a genome to get coordinate information."
            regions = pybedtools.BedTool(bedFname)
            with pysam.FastaFile(genomeFname) as genome:
                addCoordsInfo(regions, self._fp, genome)
        # We don't know the output length yet, since the model hasn't run any batches.
        # We'll construct the datasets on the fly once we get our first output.

    def buildDatasets(self, sampleOutputs: list) -> None:
        """Actually construct the output hdf5 file.

        You must give this function the first prediction from the model so that
        it can size its datasets appropriately.

        :param sampleOutputs: An output from the Batcher. This is not written to the file,
            it's just used to get the right size for the datasets.
        """
        # Since descriptions will not consume an inordinate amount of memory, and string
        # handling is messy with h5py, just store all the descriptions in a list and
        # write them out at the end.
        self._descriptionList = []
        self.headBuffers = []
        # h5 files are very slow if you write many times to non-chunked datasets.
        # So I create chunked datasets, and then create internal buffers to store
        # up to 100 entries before actually committing to the hdf5 file.
        # This optimization means that the program is now GPU-limited and not
        # h5py-limited, which is how things should be.
        for headID in range(self.numHeads):
            headGroup = self._fp.create_group(f"head_{headID}")
            # These are the storage buffers for incoming data.
            headBuffer = [np.empty((self.writeChunkSize, ), dtype=LOGCOUNT_T),  # counts
                          np.empty((self.writeChunkSize, ) + sampleOutputs[headID].shape,
                                   dtype=LOGIT_T)]  # profile
            self.headBuffers.append(headBuffer)
            headGroup.create_dataset("logcounts", (self.numPredictions,),
                                     dtype=LOGCOUNT_T,
                                     chunks=(min(self.writeChunkSize, self.numPredictions),))
            headGroup.create_dataset("logits",
                                     ((self.numPredictions,) + sampleOutputs[headID].shape),
                                     dtype=LOGIT_T,
                                     chunks=(min(self.writeChunkSize, self.numPredictions),)
                                            + sampleOutputs[headID].shape)  # noqa
        logUtils.debug("Initialized datasets.")

    def addEntry(self, batcherOut: tuple) -> None:
        """Add a single output from the Batcher."""
        # Give this exactly the output from the batcher, and it will queue the data
        # to be written to the hdf5 on the next commit.

        logitsLogcounts, label = batcherOut
        if self.writeHead == 0 and self.batchWriteHead == 0:
            # We haven't constructed our datasets yet. Do so now, because
            # now we know the output size of the model.
            self.buildDatasets(logitsLogcounts)

        self._descriptionList.append(label)

        for headID in range(self.numHeads):
            logits = logitsLogcounts[headID]
            logcounts = logitsLogcounts[headID + self.numHeads]
            self.headBuffers[headID][0][self.batchWriteHead] = logcounts
            self.headBuffers[headID][1][self.batchWriteHead] = logits
        self.batchWriteHead += 1
        # Have we filled our storage buffers? If so, write them out.
        if self.batchWriteHead == self.writeChunkSize:
            self.commit()

    def commit(self) -> None:
        """Actually write the data out to the backing hdf5 file."""
        start = self.writeHead
        stop = start + self.batchWriteHead
        for headID in range(self.numHeads):
            headGroup = self._fp[f"head_{headID}"]
            headBuffer = self.headBuffers[headID]
            headGroup["logits"][start:stop] = headBuffer[1][:self.batchWriteHead]
            headGroup["logcounts"][start:stop] = headBuffer[0][:self.batchWriteHead]
        self.writeHead += self.batchWriteHead
        self.batchWriteHead = 0

    def close(self) -> None:
        """Close the output hdf5.

        You MUST call close on this object, as otherwise the last bit of data won't
        get written to disk.
        """
        if self.batchWriteHead != 0:
            self.commit()
        stringDType = h5py.string_dtype(encoding="utf-8")
        self._fp.create_dataset("descriptions", dtype=stringDType, data=self._descriptionList)
        logUtils.info("Closing h5.")
        self._fp.close()


def addGenomeInfo(outFile: h5py.File, genome: pysam.FastaFile) -> \
        tuple[type, type, dict[str, int]]:
    """Create a chrom name and chrom size dataset so that this h5 can be converted into a bigwig.

    :param outFile: The (opened) hdf5 file to write.
    :param genome: The (opened) FastaFile object containing genome information.
    :return: The types you need to use to store chromosome index and position information,
        as well as a dictionary to map chromosome name to index.
        The first is what you need for coords_chrom, the second for coords_start and coords_end.
        They will each be one of ``np.uint8``, ``np.uint16``, ``np.uint32``, or ``np.uint64``.
        The dictionary (the third element) maps string chromosome names (like ``chrII``) to
        integers. It is the inverse of the created ``chrom_names`` dataset.
    """
    stringDtype = h5py.string_dtype(encoding="utf-8")
    outFile.create_dataset("chrom_names", (genome.nreferences,), dtype=stringDtype)
    outFile.create_dataset("chrom_sizes", (genome.nreferences,), dtype="u8")
    chromNameToIndex = {}
    chromDtype = np.uint8
    if genome.nreferences > 127:  # type: ignore
        # We could store up to 255 in a uint8, but people might
        # .astype(int8) and that would be a problem. So we sacrifice a
        # bit if there are 128 to 255 chromosomes.
        chromDtype = np.uint16
    assert len(genome.references) < 65535, "The genome has more than 2^16 chromosomes, "\
                                           "and cannot be saved using the current hdf5 "\
                                           "format. Increase the width of the coords_chrom "\
                                           "dataset to fix. Alternatively, consider predicting "\
                                           "from a fasta file, which lets you use arbitrary "\
                                           "names for each sequence."
    chromPosDtype = np.uint32
    for i, chromName in enumerate(genome.references):
        outFile["chrom_names"][i] = chromName
        chromNameToIndex[chromName] = i
        refLen = genome.get_reference_length(chromName)
        if refLen > (2 ** 31 - 1):
            logUtils.debug("The genome contains a chromosome that is over four billion bases long. "
                           "Using an 8-byte integer for chromosome positions.")
            chromPosDtype = np.uint64
        outFile["chrom_sizes"][i] = genome.get_reference_length(chromName)
    return chromDtype, chromPosDtype, chromNameToIndex


def addCoordsInfo(regions: pybedtools.BedTool, outFile: h5py.File,
                  genome: pysam.FastaFile, stopName: str = "coords_stop") -> None:
    """Initialize an hdf5 with coordinate information.

    Creates the chrom_names, chrom_sizes, coords_chrom, coords_start,
    and coords_stop datasets.

    :param regions: A BedTool of regions that will be written.
    :param outFile: The opened hdf5 file.
    :param genome: An opened pysam.FastaFile with your genome.
    :param stopName: What should the stop point dataset be called?
        For interpretation scores, it should be called coords_end, while
        for predictions it should be called coords_stop.
        I'm sorry that this parameter exists.

    """
    chromDtype, chromPosDtype, chromNameToIndex = addGenomeInfo(outFile, genome)
    logUtils.debug("Genome info datasets created. Populating regions.")
    # Build a table of chromosome numbers. For space savings, only store the
    # index into the chrom_names table.
    chromDset = [chromNameToIndex[r.chrom] for r in regions]
    startDset = [r.start for r in regions]
    stopDset = [r.stop for r in regions]
    logUtils.debug("Writing coords_chrom")
    outFile.create_dataset("coords_chrom", dtype=chromDtype, data=chromDset)
    logUtils.debug("Writing coords_start")
    outFile.create_dataset("coords_start", dtype=chromPosDtype, data=startDset)
    logUtils.debug("Writing coords_end")
    outFile.create_dataset(stopName, dtype=chromPosDtype, data=stopDset)
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
