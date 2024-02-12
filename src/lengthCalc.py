#!/usr/bin/env python3
"""A utility to calculate the input and output length of BPReveal models."""
import argparse


def getParser() -> argparse.ArgumentParser:
    """Command line arguments for the length_calc script."""
    parser = argparse.ArgumentParser(description="Calculates the input sequence "
        "length required to run BPReveal for a given configuration and output width. "
        "The required length is written to stdout.")

    parser.add_argument("--output-len",
            help="The width of the output profile that is to be predicted. If "
                 "specified, the returned value will be the required input "
                 "sequence length. Exactly one of --input-len or --output-len "
                 "must be specified.",
            dest="outputLen",
            metavar="OL",
            type=int)

    parser.add_argument("--input-len",
            help="The width of the input sequence for which profiles will be "
                 "predicted. If specified, the returned value is the width of "
                 "the predicted profile. If no profile can be predicted, the "
                 "program will print a negative number and throw an error. "
                 "Exactly one of --input-len or --output-len must be specified.",
            dest="inputLen",
            metavar="IL",
            type=int)

    parser.add_argument("--n-dil-layers",
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

    parser.add_argument("--conv1-kernel-size",
            help="The size of the first convolution in the network, typically on the order of 25",
            dest="conv1KernelSize",
            type=int,
            metavar="C1KS",
            required=False)

    parser.add_argument("--profile-kernel-size",
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


def getLengthDifference(numDilLayers, initialConvolutionWidths, profileKernelSize, verbose):
    """Determine the padding on each size of the output.

    Given a BPNet architecture, calculate how much longer the input sequence will be than
    the predicted profile.

    :param numDilLayers: The number of dilated convolutional layers in BPNet.
    :param initialConvolutionWidths: The widths of the convolutional kernels preceding the
        dilated convolutions. In the original BPNet, this was a single layer of width
        25, so this argument would be [25].
    :param profileKernelSize: The width of the final kernel that generates the profiles,
        typically around 75.
    :return: An integer representing the number of extra bases in the input compared to the width of
        the predicted profile. Divide by two to get the overhang on each side of the input.
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
            print("Convolutional layer, width {0:d}, receptive field {1:d}"
                  .format(convWidth, overhang + 1))

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
        print("After dilated convolutions, receptive field {0:d}".format(overhang + 1))
    # Now, at the bottom, we have the output filter. It's the same math as the first filter.
    assert profileKernelSize % 2 == 1
    overhang += (profileKernelSize - 1)
    if verbose:
        print("After final convolution, receptive field {0:d}".format(overhang + 1))
    return overhang


def getOutputLength(seqLen, numDilLayers, initialConvolutionWidths,
                    profileKernelSize, verbose):
    """Determine how long the output will be.

    Given a BPNet architecture and a length of the input sequence, calculate the width of the
    predicted profile.

    :param seqLen: : The length of the input sequence, in bp.
    :param numDilLayers: : The number of dilated convolutional layers in BPNet.
    :param initialConvolutionWidths: : The widths of the convolutional kernels preceding
        the dilated convolutions. In the original BPNet, this was a single layer of
        width 25, so this argument would be [25].
    :param profileKernelSize: : The width of the final kernel that generates the profiles,
        typically around 75.
    :return: An integer representing the width of the profile that will be calculated. If this
        value is zero or lower, then no bases will have their profile predicted and the
        model is invalid.
    """
    return seqLen \
        - getLengthDifference(numDilLayers, initialConvolutionWidths,
                              profileKernelSize, verbose)


def getInputLength(outPredLen, numDilLayers, initialConvolutionWidths,
                   profileKernelSize, verbose):
    """Given an output length, calculate input length.

    Given a BPNet architecture and a length of the output profile, calculate the width of the
    input sequence necessary to get that profile..

    :param outPredLen: : The length of the output profile, in bp.
    :param numDilLayers: : The number of dilated convolutional layers in BPNet.
    :param initialConvolutionWidths: : The widths of the convolutional kernels preceding
        the dilated convolutions. In the original BPNet, this was a single layer of
        width 25, so this argument would be [25].
    :param profileKernelSize: : The width of the final kernel that generates the profiles,
        typically around 75.
    :return: An integer representing the width of the sequence necessary to calculate the profile.
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
        assert outLen > 0, "Predicted output length {0:d} is empty.".format(outLen)
    else:
        assert False, "Must provide one of --input-len or --output-len"


if __name__ == "__main__":
    lengthCalcMain()
