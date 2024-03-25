#!/usr/bin/env python3
"""A script to generate importance scores in the style of the original BPNet.

BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/interpretFlat.bnf


Parameter Notes
---------------

genome, bed-file
    If you specify these two parameters in the configuration, then this program
    will read coordinates from the bed file, extract the sequences from the
    provided fasta, and run interpretations on those sequences. In this case,
    the output file will include ``chrom_names``, ``chrom_sizes``, ``coords_start``,
    ``coords_end``, and ``coords_chrom``. The ``bed-file`` that you give should have
    regions matching the model's *output* length. The regions will be
    automatically inflated in order to extract the input sequence from the
    genome. Somewhat confusingly, this means that the contribution scores will
    include contributions from bases that are not in your bed file. This is
    because the contribution scores explain how all of the input bases
    contribute to the output observed at the region in the bed file.

fasta-file
    If you specify a fasta file, then the sequences are taken directly from
    that file. In this case, the output hdf5 file will not include the
    ``chrom_names``, ``chrom_sizes``, ``coords_start``, ``coords_end``, or
    ``coords_chrom`` fields. Instead it will contain a ``descriptions`` dataset,
    which holds the description lines from the fasta. If you specify
    ``fasta-file``, then the sequences in that fasta must be as long as the
    model's *input* length. (Since we need the whole sequence that will be
    explained.) In this case, the contribution scores in the output will match
    one-to-one with the input bases.

coordinates, genome, bed-file
    If you give a fasta file and also include a ``coordinates`` section, then
    the sequences to interpret will be drawn from the given ``fasta-file``, but
    coordinate information and chromosome sizes will be taken from the bed file
    and genome fasta. This means that you can use
    :py:mod:`shapToBigwig<bpreveal.shapToBigwig>` even though the sequences
    don't come from a real genome. In this case, the output hdf5 will contain
    all of the usual coordinate datasets in addition to the ``description``
    dataset that you usually get for interpreting from a fasta file.

heads
    This parameter is the *total* number of heads that the model has.

head-id
    This parameter gives which head you want importance values calculated for.

profile-task-ids
    Lists which of the profile predictions (i.e., tasks) from the specified
    head you want considered. Almost always, you should include all of the
    profiles. For a single-task head, this would be ``[0]``, and for a two-task
    head this would be ``[0,1]``.

profile-h5, counts-h5
    These are the names of the output files that will be saved to disk.

num-shuffles
    This is the number of background samples that should be used for
    calculating shap values. I recommend 20.

kmer-size
    (Optional) If provided, this changes how the shuffles work. By default (or
    if you specify ``kmer-size = 1``) all of the bases in the input are jumbled
    randomly. However, if you specify ``kmer-size=2``, then the distribution of
    dimers will be preserved in the shuffled sequences. If you specify
    ``kmer-size=3``, then trimers will be preserved, and so on.

Output Specification
--------------------

Genome and bed file
^^^^^^^^^^^^^^^^^^^
If you gave a genome fasta and a bed file of regions, the output will have this
structure:

chrom_names
    A list of strings giving the name of each chromosome. ``coords_chrom``
    entries correspond to the order of chromosomes in this dataset.

chrom_sizes
    A list of integers giving the size of each chromosome. This is mostly here
    as a handy reference when you want to make a bigwig file.

coords_start
    The start point for each of the regions that were explained. This will have
    shape ``(num-regions,)``. Note that this starts at the beginning of the
    *input* to the model, so it will not match the coordinates in the bed file.

coords_end
    The end point for each of the regions that were explained. This will have
    shape ``(num-regions,)``. As with ``coords_start``, this corresponds to the
    last base in the *input* to the model.

coords_chrom
    The chromosome number on which each region is found. These are integer
    indexes into ``chrom_names``, and this dataset has shape ``(num-regions,)``

input_seqs
    A one-hot encoded array representing the input sequences. It will have
    shape ``(num-regions x input-length x 4)``

hyp_scores
    A table of the shap scores. It will have shape ``(num-regions x
    input-length x 4)``. If you want the actual contribution scores, not the
    hypothetical ones, multiply ``hyp_scores`` by ``input_seqs`` to zero out
    all purely hypothetical contribution scores.


Fasta file
^^^^^^^^^^

descriptions
    A list of strings that are the description lines from the input fasta file
    (with the leading ``>`` removed). This list will have shape
    ``(num-regions,)``

input_seqs, hyp_scores
    These have the same meaning as in the bed-and-genome based output files.

Additional Information
----------------------

No fasta coordinate data
^^^^^^^^^^^^^^^^^^^^^^^^

While you can use :py:mod:`shapToNumpy<bpreveal.shapToNumpy>` on either format of
``interpretFlat`` output, you cannot convert a fasta-based interpretation
h5 to a bigwig, since it doesn't contain coordinate information. You can get
around this limitation by providing a bed file and a genome in a ``coordinates``
section.

History
-------

Before BPReveal 4.0.0, the ``coords_chrom`` dataset in the generated hdf5 file
contained strings. For consistency with every other tool in the BPReveal suite,
it was changed to contain an integer index into the ``chrom_names`` dataset.

API
---


"""
import json
import h5py
import pybedtools
import pysam
from bpreveal import interpretUtils
from bpreveal import logUtils
import bpreveal.internal.disableTensorflowLogging  # pylint: disable=unused-import # noqa
from bpreveal.internal import predictUtils


def main(config: dict):
    """Run the interpretation.

    :param config: A JSON object matching the interpretFlat specification.
    """
    logUtils.setVerbosity(config["verbosity"])
    genomeFname = None
    kmerSize = 1
    if "kmer-size" in config:
        kmerSize = config["kmer-size"]
    else:
        logUtils.info("Did not find a kmer-size property in config. "
                     "Using default value of 1.")

    if "bed-file" in config:
        logUtils.debug("Configuration specifies a bed file.")
        genomeFname = config["genome"]
        generator = interpretUtils.FlatBedGenerator(bedFname=config["bed-file"],
                                                    genomeFname=genomeFname,
                                                    inputLength=config["input-length"],
                                                    outputLength=config["output-length"])
    else:
        logUtils.debug("Configuration specifies a fasta file.")
        generator = interpretUtils.FastaGenerator(config["fasta-file"])

    profileWriter = interpretUtils.FlatH5Saver(
        outputFname=config["profile-h5"], numSamples=generator.numRegions,
        inputLength=config["input-length"], genome=genomeFname,
        useTqdm=logUtils.getVerbosity() <= logUtils.INFO)
    countsWriter = interpretUtils.FlatH5Saver(
        outputFname=config["counts-h5"], numSamples=generator.numRegions,
        inputLength=config["input-length"], genome=genomeFname,
        useTqdm=False)

    batcher = interpretUtils.FlatRunner(
        modelFname=config["model-file"], headID=config["head-id"],
        numHeads=config["heads"], taskIDs=config["profile-task-ids"],
        batchSize=10, generator=generator, profileSaver=profileWriter,
        countsSaver=countsWriter, numShuffles=config["num-shuffles"],
        kmerSize=kmerSize)
    batcher.run()

    # Finishing touch - if someone gave coordinate information, load that.
    if "coordinates" in config:
        for ftype in ("profile-h5", "counts-h5"):
            with h5py.File(config[ftype], "r+") as h5fp, \
                 pysam.FastaFile(config["coordinates"]["genome"]) as genome:
                bedFp = pybedtools.BedTool(config["coordinates"]["bed-file"])
                predictUtils.addCoordsInfo(regions=bedFp, outFile=h5fp,
                                           genome=genome, stopName="coords_end")


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "r") as configFp:
        configJson = json.load(configFp)
    import bpreveal.schema
    bpreveal.schema.interpretFlat.validate(configJson)
    main(configJson)
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
