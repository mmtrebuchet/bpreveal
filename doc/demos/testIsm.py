#!/usr/bin/env python3
from collections.abc import Callable
from typing import Any
from bpreveal.internal import interpretUtils
from bpreveal import logUtils
from bpreveal import utils
from bpreveal.internal import interpreter
from bpreveal.internal.constants import ONEHOT_AR_T
import numpy as np


def minmaxMetric(headID: int, taskID: int) -> Callable:
    def metric(model: Any) -> Any:
        from keras import ops
        profileOutput = model.outputs[headID][:, :, taskID]
        minValue = ops.min(profileOutput, axis=1)
        maxValue = ops.max(profileOutput, axis=1)
        Δ = ops.subtract(maxValue, minValue)
        return Δ
    return metric

def makeShuffler(startPt, stopPt):
    def shuffler(seq: ONEHOT_AR_T, pos: int) -> list[ONEHOT_AR_T]:
        if pos < startPt or pos > stopPt:
            # Don't bother explaining bases that are way far away from the output.
            return []
        seqs = [np.copy(seq) for _ in range(4)]
        [np.random.shuffle(s) for s in seqs]
        ret = []
        retStrs = [utils.oneHotDecode(seq)]
        for shuffleSeq in seqs:
            shufStr = utils.oneHotDecode(shuffleSeq)
            if shufStr not in retStrs:
                # We haven't seen this shuffle yet.
                retStrs.append(shufStr)
                ret.append(shuffleSeq)
        return ret
        # Another shuffler: This one is for a kmerSize of 1 base,
        # and it returns all bases except the one passed in.
        # possibleRets = utils.oneHotEncode("ACGT")
        # return possibleRets[base[0] == 0]
    return shuffler


def main(config: dict) -> None:
    """Run the interpretation.

    :param config: A JSON object matching the interpretFlat specification.
    """
    logUtils.setVerbosity(config["verbosity"])
    kmerSize = 11  # Use a wide shuffle window of 11 bases.
    if "kmer-size" in config:
        kmerSize = config["kmer-size"]
    else:
        logUtils.info("Did not find a kmer-size property in config. "
                      "Using default value of 11.")

    generator = interpretUtils.FlatBedGenerator(
        bedFname=config["bed-file"],
        genomeFname=config["genome"],
        inputLength=config["input-length"],
        outputLength=config["output-length"])

    saver = interpretUtils.FlatH5Saver(
        outputFname=config["profile-h5"], numSamples=generator.numRegions,
        inputLength=config["input-length"], genome=config["genome"],
        useTqdm=True,
        config=str(config))
    metric = minmaxMetric(
        config["head-id"], config["profile-task-ids"][0])
    buffer = (config["output-length"] - config["input-length"]) // 2
    # numShuffles is irrelevant for the ism backend, so I've set it to zero here.
    batcher = interpretUtils.InterpRunner(
        modelFname=config["model-file"], metrics=[metric],
        batchSize=16, generator=generator, savers=[saver],
        numShuffles=0, kmerSize=kmerSize,
        numThreads=3, backend="ism",
        shuffler=makeShuffler(buffer, config["output-length"] - buffer)
    )
    batcher.run()


if __name__ == "__main__":
    import sys
    configJson = interpreter.evalFile(sys.argv[1])
    assert isinstance(configJson, dict)
    main(configJson)
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
