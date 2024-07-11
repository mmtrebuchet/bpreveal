#!/usr/bin/env python3
"""A little utility to plot loss values during training."""
import json
import re
import math
import datetime
import argparse
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from bpreveal import logUtils
import bpreveal

plt.rcParams["font.size"] = 8


def getParser() -> argparse.ArgumentParser:
    """Generate the argument parser."""
    parser = argparse.ArgumentParser(description="Takes in a model history file (json format) "
                                                 "and generates plots of all the components of "
                                                 "the loss during training.")
    parser.add_argument("--json",
            help="The name of the json-format model history file.")
    parser.add_argument("--output",
            help="The name of the png-format image file that should be written.")
    parser.add_argument("--dpi",
            help="(optional) The resolution of the image that should be saved.",
            default=300,
            type=int)
    parser.add_argument("--exclude",
            help="(optional) Don't include loss plots from these components of the loss. "
                "Specified as a regex, may be specified multiple times.",
            nargs="+",
            type=str)
    parser.add_argument("--start-from",
            help="(optional) Instead of starting at epoch 0, start at epoch N",
            type=int,
            default=0,
            dest="startFrom")
    parser.add_argument("--verbose",
            help="Display extra information as the losses are being processed.",
            action="store_true")
    parser.add_argument("--total-only",
            help="(optional) Instead of plotting all the loss components, "
                 "just show the total loss.",
            action="store_true")
    return parser


def reweightCountsLosses(history: dict, lossTypes: list[list[str]]) -> list[float]:
    """Add corrections for the adaptive counts loss algorithm.

    :param history: The loss history, straight from the json.
    :param lossTypes: The losses that should be plotted.
    :return: The weight history, and also edits history and lossTypes.
    """
    totalReweightedValLosses = np.zeros((len(history["loss"]),))
    totalReweightedLosses = np.zeros((len(history["loss"]),))
    countsLossWeight = history["counts-loss-weight"]
    logUtils.info("loss types before reweighting: " + str(lossTypes))
    weightLossTypes = []
    # Now we need to go back and calculate all the corrected counts losses.
    for countsKey in countsLossWeight.keys():
        history["cw_" + countsKey] = countsLossWeight[countsKey]
        weightLossTypes.append("cw_" + countsKey)
        # countsKey will be the head-name, we need to decorate it.
        countsRe = re.compile(f".*logcounts_{countsKey}_reweightable_mse")
        profileRe = re.compile(f".*profile_{countsKey}_multinomial_nll")
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
                elif re.search(profileRe, lossType):
                    # We found a profile loss. Add it to the totals, even though we don't
                    # need to reweight anything.

                    if lossType[:3] == "val":
                        totalReweightedValLosses += np.array(history[lossType])
                    else:
                        totalReweightedLosses += np.array(history[lossType])
            if len(typesToAdd) > 0:
                lossPair.extend(typesToAdd)
    # Find the total loss terms and add the reweighted values.
    history["rew_loss"] = totalReweightedLosses
    history["rew_val_loss"] = totalReweightedValLosses
    for lt in lossTypes:
        if lt[0] == "loss":
            lt.append("rew_loss")
            lt.append("rew_val_loss")
    lossTypes.append(weightLossTypes)
    return countsLossWeight


def main() -> None:
    """Make the plots."""
    args = getParser().parse_args()
    if args.verbose:
        logUtils.setVerbosity("INFO")
    else:
        logUtils.setVerbosity("WARNING")
    with open(args.json, "r") as fp:
        history = json.load(fp)

    if args.total_only:
        lossTypes = [["loss", "val_loss"]]
    else:
        lossTypes = []
        for lt in history.keys():
            if lt == "config":
                continue
            if lt == "counts-loss-weight":
                logUtils.info("Fount adaptive counts loss history.")
                continue
            if not re.search("val", lt):
                if ("val_" + lt) in history:  # pylint: disable=superfluous-parens
                    lossTypes.append([lt, "val_" + lt])
                else:
                    lossTypes.append([lt,])
    if "counts-loss-weight" in history:
        reweightCountsLosses(history, lossTypes)

    metadata = {"bpreveal_version": str(bpreveal.__version__),
                "created_date": str(datetime.datetime.today()),
                "json": str(args.json),
                "exclude": str(args.exclude),
                "start-from": str(args.startFrom)
                }
    fig = plotLosses(lossTypes, history, args.startFrom)
    fig.savefig(args.output, dpi=args.dpi, metadata=metadata)


def plotLosses(lossTypes: list[list[str]], history: dict, startFrom: int) -> Figure:
    """Given all the loss data, plot them in a grid."""
    logUtils.info("Loss types: " + str(lossTypes))
    logUtils.info("History types: " + str(list(history.keys())))
    # First, how many plots are needed?
    numRowsCols = math.ceil(len(lossTypes) ** 0.5)
    fig, axs = plt.subplots(nrows=numRowsCols, ncols=numRowsCols, sharex=True, figsize=(15, 15))
    epochs = range(len(history[lossTypes[0][0]]))
    for i, lt in enumerate(lossTypes):
        logUtils.info("Plotting loss type " + str(lt))
        allDats = []
        ax = axs[i // numRowsCols][i % numRowsCols]
        for loss in lt:
            ax.plot(epochs[startFrom:], history[loss][startFrom:], label=loss)
            allDats.extend(history[loss][startFrom:])
        ax.legend(prop={"size": 6})
        allDats.sort()
        ax.set_ylim(allDats[0] - 1e-6, allDats[-2] + 1e-6)
    return fig


if __name__ == "__main__":
    main()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
