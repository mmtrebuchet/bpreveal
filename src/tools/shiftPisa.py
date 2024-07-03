#!/usr/bin/env python3
"""A little tool to shift pisa values in hdf5 files."""
import argparse
import numpy as np
import numpy.typing as npt
import h5py
from bpreveal.internal.constants import IMPORTANCE_T, IMPORTANCE_AR_T, PRED_AR_T, PRED_T


def shiftPisa(dats: IMPORTANCE_AR_T, offset: int) -> IMPORTANCE_AR_T:
    """Shift a (2D) PISA array."""
    newDats = np.zeros(dats.shape, dtype=IMPORTANCE_T)
    if offset >= 0:
        newDats[:-offset, offset:] = dats[offset:, :-offset]
    else:
        newDats[-offset:, :offset] = dats[:offset, -offset:]
    return newDats


def shiftPredictions(preds: PRED_AR_T, offset: int) -> PRED_AR_T:
    """Shift an array of predictions."""
    ret = np.zeros(preds.shape, dtype=PRED_T)
    if offset >= 0:
        ret[:-offset] = preds[offset:]
    else:
        ret[-offset:] = preds[:offset]
    return ret


def getMask(dats: IMPORTANCE_AR_T, offset: int,
            coords: npt.NDArray | None, chroms: npt.NDArray | None) -> npt.NDArray:
    """After the data are shifted, which spaces in the output will still be valid?

    :param dats: The pisa data. These are just used to load the shape.
    :param offset: The offset being applied.
    :param coords: The genomic coordinates of each row in dats.
        If a coordinate discontinuity is detected, then values near that discontinuity
        are masked out.
    :param chroms: What chromosome (number) is each row in dats on?
        If the chromosome changes, then values near that discontinuity are masked out.
    :return: An array of ones and zeroes. Areas that are one are valid to use for more
        and areas with zeros are not valid pisa data after shifting.

    """
    Δ = offset
    mask = np.ones(dats.shape)
    # Zero out the edges.
    if Δ >= 0:
        mask[:, :Δ] = 0
    else:
        mask[:, Δ:] = 0
    if coords is not None and chroms is not None:
        for i in range(len(coords)):  # pylint: disable=consider-using-enumerate
            if Δ >= 0:
                if i + Δ >= len(coords):
                    # Off the end of the picture, invalid data here.
                    mask[i, :] = 0
                elif coords[i] + Δ != coords[i + Δ]:
                    # There was a coordinate shift. Data here are invalid.
                    mask[i, :] = 0
                elif chroms[i] != chroms[i + Δ]:
                    # We switched chromosome.
                    mask[i, :] = 0
            if Δ < 0:
                # Negative shift case.
                if i + Δ < 0:
                    # Off the top of the picture.
                    mask[i, :] = 0
                elif coords[i] + Δ != coords[i + Δ]:
                    mask[i, :] = 0
                elif chroms[i] != chroms[i + Δ]:
                    mask[i, :] = 0
    return mask


def doShift(pisaFnames: list[str], shifts: list[int], outFname: str) -> None:
    """Actually apply the shift.

    :param pisaFnames: The names of the hdf5-format files to shift.
    :param shifts: The integer shift amounts.
    :param outFname: The name of the (single) hdf5-format file to write.
    """
    pisaFps = [h5py.File(x, "r") for x in pisaFnames]
    with h5py.File(outFname, "w") as outFp:
        # Load up the data that doesn't change.
        for fieldName in ("chrom_names", "chrom_sizes", "coords_chrom",
                        "coords_base", "descriptions",
                        "head_id", "task_id", "sequence"):
            if fieldName in pisaFps[0]:
                fieldDset = pisaFps[0][fieldName]
                fieldAr = np.array(fieldDset)
                for fp in pisaFps[1:]:
                    otherDats = np.array(fp[fieldName])
                    assert np.array_equal(fieldAr, otherDats), \
                        f"Found different data in {fieldName}. Aborting."
                outFp.create_dataset(fieldName, data=fieldAr,
                                    compression=fieldDset.compression,
                                    chunks=fieldDset.chunks,
                                    dtype=fieldDset.dtype)

        # Now actually do the shifting work.
        shiftedDats = np.zeros(pisaFps[0]["shap"].shape)
        shiftedPreds = np.zeros(pisaFps[0]["input_predictions"].shape)
        shiftedShufPreds = np.zeros(pisaFps[0]["shuffle_predictions"].shape)
        mask = np.ones(shiftedDats.shape)
        for fp, Δ in zip(pisaFps, shifts):
            curShift = shiftPisa(fp["shap"], Δ)
            if "coords_base" in fp:
                curMask = getMask(fp["shap"], Δ, fp["coords_base"],
                                  fp["coords_chrom"])
            else:
                curMask = getMask(fp["shap"], Δ, None, None)
            curPred = shiftPredictions(fp["input_predictions"], Δ)
            curShufPred = shiftPredictions(fp["shuffle_predictions"], Δ)
            mask = mask * curMask
            shiftedDats += curShift
            shiftedPreds += curPred
            shiftedShufPreds += curShufPred

        maskedDats = shiftedDats * mask
        for i in range(mask.shape[0]):
            if np.max(mask[i]) == 0:
                shiftedPreds[i] = 0
                shiftedShufPreds[i, :] = 0
        meanedDats = maskedDats / len(pisaFps)
        meanedShiftedPreds = shiftedPreds / len(pisaFps)
        meanedShiftedShufPreds = shiftedShufPreds / len(pisaFps)

        outFp.create_dataset("shap", data=meanedDats, dtype=IMPORTANCE_T,
                            compression="gzip", chunks=pisaFps[0]["shap"].chunks)
        outFp.create_dataset("input_predictions", data=meanedShiftedPreds,
                             dtype="f4")
        outFp.create_dataset("shuffle_predictions", data=meanedShiftedShufPreds,
                             dtype="f4")


def getParser() -> argparse.ArgumentParser:
    """Generate (but don't parse_args()) the argument parser."""
    parser = argparse.ArgumentParser(
        description="Slide pisa data to turn endpoint-based pisa data into midpoints."
    )
    parser.add_argument("--pisa",
        help="The name of the file to shift. Cannot be used with --pisa5 and --pisa3")
    parser.add_argument("--pisa5", help="The first (5') pisa hdf5, for --mnase.")
    parser.add_argument("--pisa3", help="The second (3') pisa hdf5, for --mnase.")
    parser.add_argument(
        "--mnase",
        help="Perform the +79, -80 shift that is recommended for mnase",
        action="store_true")
    parser.add_argument("--amount", help="How much should the input be shifted? "
                        "Cannot be used with --pisa5, --pisa3, or --mnase.", type=int)
    parser.add_argument(
        "--out",
        help="The name of the hdf5 file to save."
    )
    return parser


def main() -> None:
    """Do the shifting."""
    args = getParser().parse_args()

    if args.mnase:
        doShift([args.pisa3, args.pisa5], [+79, -80], args.out)
    else:
        doShift([args.pisa], [args.amount], args.out)


if __name__ == "__main__":
    main()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
