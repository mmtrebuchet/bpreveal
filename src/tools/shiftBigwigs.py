#!/usr/bin/env python3
"""A little script to shift and combine bigwig tracks."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pyBigWig
import tqdm


def scanBigwigs(bwFname1: str, bwFname2: str, outFname: str) -> None:
    """Show a plot of the correlation between the two bigwigs, and save data to outFname."""
    bw1 = pyBigWig.open(bwFname1)
    bw2 = pyBigWig.open(bwFname2)
    chromName = list(bw1.chroms().keys())[0]
    dat1 = np.nan_to_num(bw1.values(chromName, 0,
                                    bw1.chroms(chromName)))[:20000000]
    dat2 = np.nan_to_num(bw2.values(chromName, 0,
                                    bw2.chroms(chromName)))[:20000000]
    ret = scipy.signal.correlate(dat2, dat1)
    xvals = range(-dat1.shape[0] + 1, dat1.shape[0])
    plt.plot(xvals, ret)
    plt.xlim(-100, 400)
    plt.grid()
    plt.show()
    with open(outFname, "w") as fp:
        for i in range(ret.shape[0] // 2 - 1000, ret.shape[0] // 2 + 1000):
            fp.write(f"{xvals[i]}\t{ret[i]}\n")


def doShift(bwFnames: list[str], shifts: list[int], outFname: str) -> None:
    """Actually shift the files."""
    bws = [pyBigWig.open(x) for x in bwFnames]
    bwOut = pyBigWig.open(outFname, "w")
    bwChroms = bws[0].chroms()
    header = [(x, bwChroms[x]) for x in sorted(bwChroms.keys())]
    bwOut.addHeader(header)
    chromDats = {}
    for chromName in tqdm.tqdm(sorted(bwChroms.keys())):
        datOut = np.zeros((bwChroms[chromName], ), dtype=np.float32)
        for i, bw in enumerate(bws):
            s = shifts[i]
            d = np.nan_to_num(bw.values(chromName, 0, bwChroms[chromName]))
            if s >= 0:
                datOut[s:] += d[:-s]
            else:
                datOut[:s] += d[-s:]
        if np.max(shifts) > 0:
            datOut[:np.max(shifts)] = 0
        if np.min(shifts) < 0:
            datOut[min(shifts):] = 0
        chromDats[chromName] = datOut
        bwOut.addEntries(chromName,
                         0,
                         values=[float(x) for x in datOut],
                         span=1,
                         step=1)
    bwOut.close()


def getParser() -> argparse.ArgumentParser:
    """Generate (but don't call parse_args()) the parser."""
    parser = argparse.ArgumentParser(
        description="Slide bigwigs to turn mnase endpoint data into midpoints."
    )
    parser.add_argument("--bw5", help="The first (5') bigwig file.")
    parser.add_argument("--bw3", help="The second (3') bigwig file.")
    parser.add_argument(
        "--scan",
        help="Measure the cross-correlation between the bigwigs and display a plot.",
        action="store_true")
    parser.add_argument(
        "--mnase",
        help="Perform the +79, -80 shift that is recommended for mnase",
        action="store_true")
    parser.add_argument(
        "--out",
        help="The name of the bigwig file to write, or, if scanning, "
             "the name of the dat file containing the correlation."
    )
    return parser


def main() -> None:
    """Run the program."""
    args = getParser().parse_args()

    if args.scan:
        scanBigwigs(args.bw5, args.bw3, args.out)
        return

    if args.mnase:
        doShift([args.bw5, args.bw3], [+79, -80], args.out)


if __name__ == "__main__":
    main()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
