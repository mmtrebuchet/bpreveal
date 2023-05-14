#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
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
        lossTypes = [('loss', 'val_loss')]
    else:
        lossTypes = []
        for lt in history.keys():
            if (not re.search('val', lt)):
                if (("val_" + lt) in history):
                    lossTypes.append((lt, "val_" + lt))
                else:
                    lossTypes.append((lt,))

    fig = plotLosses(lossTypes, history, args.startFrom)
    fig.savefig(args.output, dpi=args.dpi)


def plotLosses(lossTypes, history, startFrom):
    #First, how many plots are needed? 
    num_rowscols = math.ceil(len(lossTypes) ** 0.5)
    fig, axs = plt.subplots(nrows=num_rowscols, ncols=num_rowscols, sharex=True, figsize=(10, 10))
    epochs = range(len(history[lossTypes[0][0]]))
    for i, lt in enumerate(lossTypes):
        ax = axs[i // num_rowscols][i % num_rowscols]
        for loss in lt:
            ax.plot(epochs[startFrom:], history[loss][startFrom:], label=loss)
        ax.legend(prop={"size": 6})
    return fig


if (__name__ == "__main__"):
    main()
