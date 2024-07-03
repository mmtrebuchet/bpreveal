#!/usr/bin/env python3
"""Extract loss weights that should be used to train a model.

This is a little utility to read in the training history json and calculate
an appropriate counts loss weight parameter to get the loss balance you want.
Since we only know how large the losses for profile and counts will be after
we've trained a model, you have to use this script *after* training a burner model
in order to find appropriate settings, then use those to re-train your model.
How to use:
First, pick a counts-loss-weight that sounds reasonable, like 10. Train a model.
Run this script on the history json from that training, giving it a desired
ratio of counts to profile loss.

Note that this script is not necessary if you're using the adaptive counts
loss algorithm.

"""
# flake8: noqa: T201
import json
import re
import argparse
countsReTrain = re.compile(r"^(?P<mode>[^v].*)_logcounts_(?P<head>.*)_loss$")
profileReTrain = re.compile(r"^(?P<mode>[^v].*)_profile_(?P<head>.*)_loss$")
countsReVal = re.compile(r"^val_(?P<mode>.*)_logcounts_(?P<head>.*)_loss$")
profileReVal = re.compile(r"^val_(?P<mode>.*)_profile_(?P<head>.*)_loss$")


def loadLosses(lossStats: dict) -> dict[str, dict[str, float]]:
    """Read in the losses from the history file."""
    ret = {}
    for k in lossStats.keys():
        if m := re.match(countsReTrain, k):
            head = m.group("head")
            if head not in ret:
                ret[head] = {}
            ret[head]["countsTrain"] = lossStats[k][-1]

        if m := re.match(profileReTrain, k):
            head = m.group("head")
            if head not in ret:
                ret[head] = {}
            ret[head]["profileTrain"] = lossStats[k][-1]

        if m := re.match(countsReVal, k):
            head = m.group("head")
            if head not in ret:
                ret[head] = {}
            ret[head]["countsVal"] = lossStats[k][-1]

        if m := re.match(profileReVal, k):
            head = m.group("head")
            if head not in ret:
                ret[head] = {}
            ret[head]["profileVal"] = lossStats[k][-1]
    return ret


def addLossRatios(lossByHead: dict, targetRatio: float) -> None:
    """Adds in ratio information to the loss dict.

    :param lossByHead: The loss dict for a given head.
    :param targetRatio: The fraction of the loss that you'd like to be due to 
    """
    for k in lossByHead.keys():
        trainRatio = lossByHead[k]["countsTrain"] / lossByHead[k]["profileTrain"]
        valRatio = lossByHead[k]["countsVal"] / lossByHead[k]["profileVal"]
        lossByHead[k]["trainRatio"] = trainRatio
        lossByHead[k]["valRatio"] = valRatio
        lossByHead[k]["newWeight"] = targetRatio / valRatio


def main(jsonFname: str, targetRatio: float, prevWeight: float) -> None:
    """Read in the loss history and estimate an appropriate new weight.

    :param jsonFname: The name of the history file.
    :param targetRatio: The fraction of the loss that you'd like to be due to counts.
    :param prevWeight: The counts-loss-weight used to train the model.
    """
    with open(jsonFname, "r") as fp:
        lossStats = json.load(fp)
    lossByHead = loadLosses(lossStats)
    addLossRatios(lossByHead, targetRatio)
    if targetRatio > 0:
        print("{0:10s}\t{1:10s}\t{2:10s}\t{3:20s}"  # pylint: disable=consider-using-f-string
              .format("C/Ptrain", "C/Pval", "newWeight", "name"))
    else:
        print("{0:10s}\t{1:10s}\t{2:20s}"  # pylint: disable=consider-using-f-string
              .format("C/Ptrain", "C/Pval", "name"))
    for k, v in lossByHead.items():
        cpTrain = v["trainRatio"] * prevWeight
        cpVal = v["valRatio"] * prevWeight
        if targetRatio > 0:
            newWeight = v["newWeight"]
            print(f"{cpTrain:10f}\t{cpVal:10f}\t{newWeight:10f}\t{k:20s}")
        else:
            print(f"{cpTrain:10f}\t{cpVal:10f}\t{k:20s}")


def getParser() -> argparse.ArgumentParser:
    """Load (but don't parse_args()) the argument parser."""
    parser = argparse.ArgumentParser(description="Read in a model history json and calculate the"
                                     "profile/counts loss ratio.")
    parser.add_argument("--json", help="The name of the history json file", type=str)
    parser.add_argument("--target-ratio", help="(optional) What counts loss weight do you want "
        "to have? A float from [0,∞), with 0 meaning no counts weight, 1 meaning equal weight "
        "between counts and profile. A normal setting would be 0.1. Use the output to set "
        "counts-loss-weight in your training configuration files.",  # noqa
        dest="targetRatio", type=float, default=0.0)  # noqa
    parser.add_argument("--prev-weight", help="The counts-loss-weight you used to train your "
        "model. If provided, the printed ratios will be scaled by this number to reflect their "
        "actual contribution in your model.", type=float, default=1.0, dest="prevWeight")
    return parser


if __name__ == "__main__":
    args = getParser().parse_args()
    main(args.json, args.targetRatio, args.prevWeight)
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
