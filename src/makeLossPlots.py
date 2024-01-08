#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
import math

plt.rcParams['font.size'] = 8


def main():
    parser = argparse.ArgumentParser(description="Takes in a model history file (json format) "
                                                 "and generates plots of all the components of "
                                                 "the loss during training.")
    parser.add_argument("--json",
            help="The name of the json-format model history file.")
    parser.add_argument("--output",
            help="The name of the png-format image file that should be written.")
    parser.add_argument("--dpi",
            help='(optional) The resolution of the image that should be saved.',
            default=300,
            type=int)
    parser.add_argument("--exclude",
            help="(optional) Don't include loss plots from these components of the loss. "
                "Specified as a regex, may be specified multiple times.",
            nargs='+',
            type=str)
    parser.add_argument("--start-from",
            help='(optional) Instead of starting at epoch 0, start at epoch N',
            type=int,
            default=0,
            dest='startFrom')
    parser.add_argument("--total-only",
            help="(optional) Instead of plotting all the loss components, "
                 "just show the total loss.",
            action='store_true')
    args = parser.parse_args()

    with open(args.json, 'r') as fp:
        history = json.load(fp)

    if (args.total_only):
        lossTypes = [['loss', 'val_loss']]
    else:
        lossTypes = []
        for lt in history.keys():
            if lt == 'counts-loss-weight':
                continue
            if (not re.search('val', lt)):
                if (("val_" + lt) in history):
                    lossTypes.append([lt, "val_" + lt])
                else:
                    lossTypes.append([lt,])
    countsLossWeight = None
    lossTypesCountsWeight = []
    if "counts-loss-weight" in history:
        totalReweightedValLosses = np.zeros((len(history["loss"]),))
        totalReweightedLosses = np.zeros((len(history["loss"]),))
        countsLossWeight = history["counts-loss-weight"]
        # Now we need to go back and calculate all the corrected counts losses.
        for countsKey in countsLossWeight.keys():
            history["cw_" + countsKey] = countsLossWeight[countsKey]
            lossTypesCountsWeight.append("cw_" + countsKey)
            # countsKey will be the head-name, we need to decorate it.
            countsRe = re.compile(".*logcounts_{0:s}_loss".format(countsKey))
            profileRe = re.compile(".*profile_{0:s}_loss".format(countsKey))
            for lossPair in lossTypes:
                typesToAdd = []
                for lossType in lossPair:
                    if re.search(countsRe, lossType):
                        # We found a counts loss and know the right weights.
                        weightedCountsLosses = np.array(history[lossType])
                        weightsAr = np.array(countsLossWeight[countsKey])
                        unweightedCountsLosses = weightedCountsLosses / weightsAr
                        reweightedCountsLosses = unweightedCountsLosses * weightsAr[-1]
                        history["unw_" + lossType] = unweightedCountsLosses
                        typesToAdd.append("unw_" + lossType)
                        history["rew_" + lossType] = unweightedCountsLosses * weightsAr[-1]
                        typesToAdd.append("rew_" + lossType)
                        if lossType[:3] == "val":
                            totalReweightedValLosses += reweightedCountsLosses
                        else:
                            totalReweightedLosses += reweightedCountsLosses
                    if re.search(profileRe, lossType):
                        # We found a profile loss. Add it to the totals, even though we don't
                        # need to reweight anything.

                        if lossType[:3] == "val":
                            totalReweightedValLosses += np.array(history[lossType])
                        else:
                            totalReweightedLosses += np.array(history[lossType])
                if len(typesToAdd):
                    lossPair.extend(typesToAdd)
        # Find the total loss terms and add the reweighted values.
        history["rew_loss"] = totalReweightedLosses
        history["rew_val_loss"] = totalReweightedValLosses
        for lt in lossTypes:
            if lt[0] == "loss":
                lt.append("rew_loss")
                lt.append("rew_val_loss")
        lossTypes.append(lossTypesCountsWeight)

    fig = plotLosses(lossTypes, history, args.startFrom, countsLossWeight)
    fig.savefig(args.output, dpi=args.dpi)


def plotLosses(lossTypes, history, startFrom, countsLossWeight):
    # First, how many plots are needed?
    num_rowscols = math.ceil(len(lossTypes) ** 0.5)
    fig, axs = plt.subplots(nrows=num_rowscols, ncols=num_rowscols, sharex=True, figsize=(15, 15))
    epochs = range(len(history[lossTypes[0][0]]))
    for i, lt in enumerate(lossTypes):
        allDats = []
        ax = axs[i // num_rowscols][i % num_rowscols]
        for loss in lt:
            ax.plot(epochs[startFrom:], history[loss][startFrom:], label=loss)
            allDats.extend(history[loss][startFrom:])
        ax.legend(prop={"size": 6})
        allDats.sort()
        ax.set_ylim(allDats[0] - 1e-6, allDats[-2] + 1e-6)
    return fig


if (__name__ == "__main__"):
    main()
