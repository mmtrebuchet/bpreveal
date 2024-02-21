#!/usr/bin/env python3
# This is used by the acceptance test script to test the GA.
import argparse
from bpreveal import gaOptimize
import pysam
import tqdm
from bpreveal import utils
from bpreveal.internal import disableTensorflowLogging
del disableTensorflowLogging

utils.setVerbosity("INFO")


def runGaOptimization(predictor, fitness, originalSequence, inputLength,
                      popSize, numCorruptors, candidateCorruptors,
                      numGenerations):
    pop = gaOptimize.Population(originalSequence, inputLength, popSize,
                                numCorruptors, candidateCorruptors,
                                lambda _: True, fitness, 1, predictor)

    pop.runCalculation()

    # Now it's time to actually run the GA.
    bestScore = pop.organisms[-1].score
    pbar = tqdm.tqdm(range(numGenerations))
    for i in pbar:
        pop.nextGeneration()
        pop.runCalculation()
        # Only for showing, this is dumb to do in production.
        if pop.organisms[-1].score > bestScore:
            bestScore = pop.organisms[-1].score
            pbar.set_description("Gen {0:d} fitness {1:f}".format(
                i, bestScore))
    return pop.organisms[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="Start position for region.", type=int)
    ap.add_argument("--input-len", help="input length",
                    type=int, dest="inputLen")
    ap.add_argument("--chrom")
    ap.add_argument("--model")
    ap.add_argument("--genome")
    ap.add_argument("--output")

    args = ap.parse_args()

    outputStart = args.start
    inputLength = args.inputLen  # This is the input length of my model.
    buffer = (inputLength - 1000) // 2
    inputStart = outputStart - buffer
    inputEnd = inputStart + inputLength

    modelFname = args.model
    genomeFname = args.genome

    with pysam.FastaFile(genomeFname) as genome:
        # Read 100 extra bases because corruptors could cause deletions and we'll need
        # enough DNA sequence to feed the model in that case.
        origSequence = genome.fetch(
            args.chrom, inputStart, inputEnd + 100).upper()

    predictor = utils.BatchPredictor(modelFname, 64)
    predictor.submitString(origSequence[:inputLength], 0)
    origRet = predictor.getOutput()[0]

    def fitnessSuppressSecond(preds, _):
        # Just maximize the oct4 logcounts
        return preds[4]

    fitness = fitnessSuppressSecond

    candidateCorruptors = gaOptimize.getCandidateCorruptorList(
        origSequence,
        [(buffer, buffer + 1000)])

    bestOrganism = runGaOptimization(predictor, fitness, origSequence,
                                     inputLength, 100,
                                     10, candidateCorruptors,
                                     100)
    print(bestOrganism.corruptors)
    outDats = {
        "corruptors": str(bestOrganism.corruptors),
        "profile": [],
        "origProfile": []
    }
    for i in range(4):
        prof = utils.logitsToProfile(bestOrganism.profile[i],
                                     bestOrganism.profile[i+4])
        outDats["profile"].append(prof.tolist())
        origProf = utils.logitsToProfile(origRet[i],
                                         origRet[i+4])
        outDats["origProfile"].append(origProf.tolist())
    with open(args.output, "w") as fp:
        import json
        json.dump(outDats, fp)


if __name__ == "__main__":
    main()
