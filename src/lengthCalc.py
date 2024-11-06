#!/usr/bin/env python3
"""A utility to calculate the input and output length of BPReveal models."""
# flake8: noqa: T201
import argparse


def getParser() -> argparse.ArgumentParser:
    """Command line arguments for the length_calc script."""
    parser = argparse.ArgumentParser(description="Calculates the input sequence "
        "length required to run BPReveal for a given configuration and output length. "
        "The required length is written to stdout.")

    parser.add_argument("--output-len", "--output-length",
            help="The length of the output profile that is to be predicted. If "
                 "specified, the returned value will be the required input "
                 "sequence length. Exactly one of --input-len or --output-len "
                 "must be specified.",
            dest="outputLen",
            metavar="OL",
            type=int)

    parser.add_argument("--input-len", "--input-length",
            help="The length of the input sequence for which profiles will be "
                 "predicted. If specified, the returned value is the length of "
                 "the predicted profile. If no profile can be predicted, the "
                 "program will print a negative number and throw an error. "
                 "Exactly one of --input-len or --output-len must be specified.",
            dest="inputLen",
            metavar="IL",
            type=int)

    parser.add_argument("--n-dil-layers", "--layers",
            help="The number of diluted convolutional layers in the network, "
                 "typically on the order of 10.",
            dest="nDilLayers",
            type=int,
            metavar="NL",
            required=True)

    parser.add_argument("--initial-convolution-widths",
            help="A (space delimited) list of widths of the convolutions at the "
                 "top of the network. Mutually exclusive with --conv1-kernel-size, "
                 "since --conv1-kernel-size N is equivalent to --initial-convolution-widths N",
            dest="initialConvolutionWidths",
            type=int,
            nargs="+",
            metavar="ICW",
            required=False)

    parser.add_argument("--conv1-kernel-size", "--input-filter-width",
            help="The size of the first convolution in the network, typically on the order of 25",
            dest="conv1KernelSize",
            type=int,
            metavar="C1KS",
            required=False)

    parser.add_argument("--profile-kernel-size", "--output-filter-width",
            help="The width of the final convolutional filter in the output heads, "
                 "typically on the order of 75.",
            dest="profileKernelSize",
            type=int,
            metavar="PKS",
            required=True)
    parser.add_argument("--verbose",
            help="Display the receptive field at each level of the network",
            action="store_true",
            dest="verbose")
    return parser


def getLengthDifference(numDilLayers: int, initialConvolutionWidths: list[int],
                        profileKernelSize: int, verbose: bool):
    """Determine the padding on each size of the output.

    Given a BPNet architecture, calculate how much longer the input sequence will be than
    the predicted profile.

    :param numDilLayers: The number of dilated convolutional layers in BPNet.
    :param initialConvolutionWidths: The widths of the convolutional kernels preceding the
        dilated convolutions. In the original BPNet, this was a single layer of width
        25, so this argument would be [25].
    :param profileKernelSize: The width of the final kernel that generates the profiles,
        typically around 75.
    :param verbose: Should the code tell you what it's doing?
    :return: An integer representing the number of extra bases in the input compared to
        the length of the predicted profile. Divide by two to get the overhang on each
        side of the input.
    """
    overhang = 0
    for convWidth in initialConvolutionWidths:
        # First, remove the conv1 layer. The layer width must be odd.
        assert convWidth % 2 == 1
        # How many bases to trim? Consider a filter of width 5.
        #    DATADATADATADATADATA
        #    12345          12345
        #    ''PREDICTIONPREDIC''
        # We remove convWidth // 2 bases on each side, for a total of convWidth-1.
        overhang += (convWidth - 1)
        if verbose:
            print(f"Convolutional layer, width {convWidth}, receptive field {overhang + 1}")

    # Now we iterate through the dilated convolutions. The dilation rate starts at 2, then doubles
    # at each layer in the network.
    #     DATADATADATADATADATADATADATA
    #     C O N                  C O N
    #     ''INTERMEDIATEINTERMEDIATE''
    #       C   O   N      C   O   N
    #       ''''PREDICTIONPREDIC''''
    # The pattern is that the first layer removes four bases, the next eight bases, and so on.
    # N
    # __
    # \
    # /_  2^(i+1)  = 2^(i+2)-4
    # i=1
    overhang += 2 ** (numDilLayers + 2) - 4
    if verbose:
        print(f"After dilated convolutions, receptive field {overhang + 1}")
    # Now, at the bottom, we have the output filter. It's the same math as the first filter.
    assert profileKernelSize % 2 == 1
    overhang += (profileKernelSize - 1)
    if verbose:
        print(f"After final convolution, receptive field {overhang + 1}")
    return overhang


def getOutputLength(seqLen: int, numDilLayers: int, initialConvolutionWidths: list[int],
                    profileKernelSize: int, verbose: bool):
    """Determine how long the output will be.

    Given a BPNet architecture and a length of the input sequence, calculate the length of the
    predicted profile.

    :param seqLen: : The length of the input sequence, in bp.
    :param numDilLayers: : The number of dilated convolutional layers in BPNet.
    :param initialConvolutionWidths: : The widths of the convolutional kernels preceding
        the dilated convolutions. In the original BPNet, this was a single layer of
        width 25, so this argument would be [25].
    :param profileKernelSize: : The width of the final kernel that generates the profiles,
        typically around 75.
    :param verbose: Should the code tell you what it's doing?
    :return: An integer representing the length of the profile that will be calculated. If this
        value is zero or lower, then no bases will have their profile predicted and the
        model is invalid.
    """
    return seqLen \
        - getLengthDifference(numDilLayers, initialConvolutionWidths,
                              profileKernelSize, verbose)


def getInputLength(outPredLen: int, numDilLayers: int, initialConvolutionWidths: list[int],
                   profileKernelSize: int, verbose: bool):
    """Given an output length, calculate input length.

    Given a BPNet architecture and a length of the output profile, calculate the length
    of the input sequence necessary to get that profile..

    :param outPredLen: : The length of the output profile, in bp.
    :param numDilLayers: : The number of dilated convolutional layers in BPNet.
    :param initialConvolutionWidths: : The widths of the convolutional kernels preceding
        the dilated convolutions. In the original BPNet, this was a single layer of
        width 25, so this argument would be [25].
    :param profileKernelSize: : The width of the final kernel that generates the profiles,
        typically around 75.
    :param verbose: Should the code tell you what it's doing?
    :return: An integer representing the length of the sequence necessary to calculate the profile.
    """
    return outPredLen \
        + getLengthDifference(numDilLayers, initialConvolutionWidths,
                              profileKernelSize, verbose)


def lengthCalcMain():
    """Run the calculation."""
    parser = getParser()
    args = parser.parse_args()
    if args.conv1KernelSize is not None:
        args.initialConvolutionWidths = [args.conv1KernelSize]
    if args.initialConvolutionWidths is None:
        args.print_help()
        return
    if args.outputLen is not None:
        inpLen = getInputLength(outPredLen=args.outputLen, numDilLayers=args.nDilLayers,
                initialConvolutionWidths=args.initialConvolutionWidths,
                profileKernelSize=args.profileKernelSize,
                verbose=args.verbose)
        print(inpLen)
    elif args.inputLen is not None:
        outLen = getOutputLength(seqLen=args.inputLen, numDilLayers=args.nDilLayers,
                initialConvolutionWidths=args.initialConvolutionWidths,
                profileKernelSize=args.profileKernelSize,
                verbose=args.verbose)
        print(outLen)
        assert outLen > 0, f"Predicted output length {outLen} is empty."
    else:
        raise ValueError("Must provide one of --input-len or --output-len")


if __name__ == "__main__":
    lengthCalcMain()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
