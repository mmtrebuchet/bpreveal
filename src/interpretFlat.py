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

Shap and sequence
^^^^^^^^^^^^^^^^^

Whether you gave a fasta file or a genome and bed file, the generated hdf5 file
will contain the following datasets:

input_seqs
    A one-hot encoded array representing the input sequences. It will have
    shape ``(num-regions x input-length x NUM_BASES)``

hyp_scores
    A table of the shap scores. It will have shape ``(num-regions x
    input-length x NUM_BASES)``. If you want the actual contribution scores, not the
    hypothetical ones, multiply ``hyp_scores`` by ``input_seqs`` to zero out
    all purely hypothetical contribution scores.

input_predictions
    An array of the values of the given metric for each region that was shapped.
    This will have shape ``(num-regions,)``. (Added in BPReveal 5.1.0)

metadata
    A group containing the configuration used to generate the scores.


Genome and bed file
^^^^^^^^^^^^^^^^^^^
If you gave a genome fasta and a bed file of regions, the output will have this
structure (in addition to ``input_seqs``, ``hyp_scores``, and ``metadata``):

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


Fasta file
^^^^^^^^^^

If you provided a fasta file for sequence input, then the following
dataset will be created:

descriptions
    A list of strings that are the description lines from the input fasta file
    (with the leading ``>`` removed). This list will have shape
    ``(num-regions,)``

Additional Information
----------------------

No fasta coordinate data
^^^^^^^^^^^^^^^^^^^^^^^^

While you can use :py:mod:`shapToNumpy<bpreveal.shapToNumpy>` on either format of
``interpretFlat`` output, you cannot convert a fasta-based interpretation
h5 to a bigwig, since it doesn't contain coordinate information. You can get
around this limitation by providing a bed file and a genome in a ``coordinates``
section.

Using custom metrics or the ISM backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This script is hard-wired to use the normal profile and counts metric, and it calculates
shap scores. If you want to use the flexibility of the interpretation machinery for your
own purposes, you can check out a sample script that uses a custom metric as well as the
ISM backend, located at ``doc/demos/testIsm.py``.

History
-------

Before BPReveal 4.0.0, the ``coords_chrom`` dataset in the generated hdf5 file
contained strings. For consistency with every other tool in the BPReveal suite,
it was changed to contain an integer index into the ``chrom_names`` dataset.

The ``input_predictions`` field was added in BPReveal 5.1.0.

API
---


"""
import sys
from collections.abc import Callable
from typing import Any
import h5py
import pybedtools
import pysam
import bpreveal.schema
from bpreveal.internal import interpretUtils
from bpreveal import logUtils
from bpreveal.internal import predictUtils
from bpreveal.internal import interpreter


def profileMetric(headID: int, taskIDs: list[int]) -> Callable:
    """A metric to extract the profile spikiness from a given head and list of tasks.

    :param headID: The head number that you want counts for.
    :param taskIDs: The tasks within this head that you want included in the metric.
        For most uses, this would be all of the tasks. For example, a two-task head would
        use ``taskIDs = [0,1]``.
    :return: A function (that takes a model as its argument) that can
        be passed into the depths of the interpretation system.
    """
    def metric(model: Any) -> Any:
        """The metric to evaluate.

        :param model: The BPReveal model to use.
        :return: A scalar Tensor that can be shapped back.
        """
        # Note that keras and tensorflow must be imported INSIDE the returned function so
        # that they haven't been imported when the interpretation machinery tries to start
        # up. This is dumb but it's how Tensorflow works.
        import keras
        import tensorflow as tf
        from keras import ops

        # pylint: disable=abstract-method,too-many-ancestors,arguments-differ
        class StopGradLayer(keras.layers.Layer):
            """Because the Tensorflow 2.16 upgrade wasn't painful enough...

            This just wraps stop_gradient so that it can be called
            with a KerasTensor.
            """

            def call(self, x: tf.Tensor) -> keras.KerasTensor:
                """Actually stop the gradient.

                :param x: the input tensor that shouldn't be backpropped.
                :return: A new tensor.
                """
                return ops.stop_gradient(x)
        # pylint: enable=abstract-method,too-many-ancestors,arguments-differ
        # The profile logits for the given head. Has shape (batch-size, output-length, num-tasks)
        profileOutput = model.outputs[headID]
        # Select only the task IDs that were selected.
        # This has shape (batch-size, output-length, num-selected-tasks)
        stackedLogits = ops.stack([profileOutput[:, :, x] for x in taskIDs], axis=2)
        inputShape = stackedLogits.shape
        # Flatten all of the logits into a vector of shape (output-length * num-selected-tasks,)
        numSamples = inputShape[1] * inputShape[2]
        logits = ops.reshape(stackedLogits, [-1, numSamples])
        meannormedLogits = ops.subtract(logits, ops.mean(logits, axis=1)[:, None])
        # We don't want to propagate the shap calculation through this layer.
        # Note that stopgrad is meaningless when using the ism backend.
        stopgradMeannormedLogits = StopGradLayer()(meannormedLogits)
        # The axis=1 here is because we don't want to combine samples from
        # different batches (and batches are in axis=0).
        softmaxOut = ops.softmax(stopgradMeannormedLogits, axis=1)
        weightedSum = ops.sum(softmaxOut * meannormedLogits, axis=1)
        return weightedSum
    # We return the *function* that accepts a model and returns the
    # metric.
    return metric


def countsMetric(numHeads: int, headID: int) -> Callable:
    """A metric to extract the logcounts from the given head.

    :param numHeads: The total number of heads that the model will have.
    :param headID: The head number that you want counts for.
    :return: A function (that takes a model as its argument) that can
        be passed into the depths of the interpretation system.
    """
    # Note that this slice captures the whole batch, hence the colon
    # on axis 0 of the model's output head.
    return lambda model: model.outputs[numHeads + headID][:, 0]


def interpretFlat(config: dict) -> None:
    """Run the interpretation.

    :param config: A JSON object matching the interpretFlat specification.
    """
    logUtils.setVerbosity(config["verbosity"])
    genomeFname = None
    kmerSize = 1
    logUtils.info(f"InterpretFlat starting with model file {config["model-file"]}")
    logUtils.info(f"Using head ID {config["head-id"]}")
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
        logUtils.info(f"Created a reader for bed file {config["bed-file"]}")
    else:
        logUtils.debug("Configuration specifies a fasta file.")
        generator = interpretUtils.FastaGenerator(config["fasta-file"])
        logUtils.info(f"Created a reader for fasta file {config["fasta-file"]}")
    writers = []
    metrics = []
    if "profile-h5" in config:
        logUtils.debug(f"Initializing profile saver for {config["profile-h5"]}.")
        profileWriter = interpretUtils.FlatH5Saver(
            outputFname=config["profile-h5"], numSamples=generator.numRegions,
            inputLength=config["input-length"], genome=genomeFname,
            useTqdm=logUtils.getVerbosity() <= logUtils.INFO,
            config=str(config))
        writers.append(profileWriter)
        profileMetricFun = profileMetric(config["head-id"], config["profile-task-ids"])
        metrics.append(profileMetricFun)
    if "counts-h5" in config:
        logUtils.debug(f"Initializing counts saver for {config["profile-h5"]}.")
        countsWriter = interpretUtils.FlatH5Saver(
            outputFname=config["counts-h5"], numSamples=generator.numRegions,
            inputLength=config["input-length"], genome=genomeFname,
            useTqdm=False,
            config=str(config))
        writers.append(countsWriter)
        countsMetricFun = countsMetric(config["heads"], config["head-id"])
        metrics.append(countsMetricFun)
    if len(metrics) != 2:
        logUtils.warning(f"Initialized {len(metrics)} metrics instead of the expected two.")
        logUtils.warning("Did you forget to include both profile-h5 and counts-h5 in your")
        logUtils.warning("configuration file?")
        logUtils.warning("interpretFlat will only run one metric.")
        logUtils.warning("This was added in 5.2.1 and will no longer be a warning in 5.3.0.")
    assert len(metrics) > 0, "You must specify at least one of profile-h5 or counts-h5."
    # If you want to use a custom metric, you could add that here.
    # Remember that numThreads must be divisible by the number of metrics.
    batcher = interpretUtils.InterpRunner(
        modelFname=config["model-file"], metrics=metrics,
        batchSize=10, generator=generator, savers=writers,
        numShuffles=config["num-shuffles"], kmerSize=kmerSize, numThreads=2,
        backend="shap",
        useHypotheticalContribs=True)
    batcher.run()

    # Finishing touch - if someone gave coordinate information, load that.
    if "coordinates" in config:
        for ftype in ("profile-h5", "counts-h5"):
            if ftype in config:
                with h5py.File(config[ftype], "r+") as h5fp, \
                        pysam.FastaFile(config["coordinates"]["genome"]) as genome:
                    bedFp = pybedtools.BedTool(config["coordinates"]["bed-file"])
                    predictUtils.addCoordsInfo(regions=bedFp, outFile=h5fp,
                                               genome=genome, stopName="coords_end")
    logUtils.info("interpretFlat complete, exiting.")


def main() -> None:
    """A zero-argument wrapper around the main function."""
    configJson = interpreter.evalFile(sys.argv[1])
    assert isinstance(configJson, dict)
    bpreveal.schema.interpretFlat.validate(configJson)
    interpretFlat(configJson)


if __name__ == "__main__":
    main()
# Copyright 2022-2025 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
