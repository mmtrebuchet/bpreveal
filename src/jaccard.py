"""Thin wrapper for the internal libjaccard C library."""
import bpreveal.internal.libjaccard as lj
from bpreveal.internal import constants


def slidingJaccard(importanceScores: constants.IMPORTANCE_AR_T,
                   cwm: constants.MOTIF_FLOAT_AR_T) -> \
        tuple[constants.MOTIF_FLOAT_AR_T, constants.MOTIF_FLOAT_AR_T]:
    """Calculate the sliding Jaccard similarity.

    :param importanceScores: An array of shape (M, NUM_BASES) giving hypothetical importance
        scores.
    :param cwm: An array of shape (N, NUM_BASES), giving a motif's CWM.
    :return: A tuple of arrays, both with shape (M - N + 1).
        The first one gives the sliding Jaccard similarities
        and the second gives the contribution magnitudes.

    The returned array is defined by::
        slidingJaccard(A,B)[i] = jaccardRegion(A[i:i+N],B)
    """
    return lj.slidingJaccard(importanceScores, cwm)


def jaccardRegion(importanceScores: constants.IMPORTANCE_AR_T,
                  scaleFactor: float, cwm: constants.MOTIF_FLOAT_AR_T) -> float:
    r"""For given region's importance scores, calculate the continuous Jaccard similarity.

    :param importanceScores: An array of shape (length, NUM_BASES)
        giving a region's hypothetical importance scores.
    :param scaleFactor: A constant that the importance scores should be multiplied by.
    :param cwm: An array of shape (length, NUM_BASES) giving the CWM for a motif.
    :return: A single float giving the Jaccard match

    This implements the formula in the modisco paper, namely that

    #ifdef MAN_PAGE
    .. code-block :: text

                   Σ (v[1,i] ∩ v[2,i])
        J(v₁,v₂) = -------------------
                   Σ (v[1,i] ∪ v[2,i])

    where:

    .. code-block :: text

        x ∩ y = min(|x|, |y|) * sign(x) * sign(y)
        x ∪ y = max(|x|, |y|)
    #else
    .. math::

        J(v_1,v_2) = \frac{\sum_i (v_{1,i} \cap v_{2,i})}{\sum_i (v_{1,i} \cup v_{2,i})}

    where:

    .. math::

        x \cap y &= min(|x|, |y|) * sign(x) * sign(y) \\
        x \cup y &= max(|x|, |y|)

    #endif


    The scaleFactor is a number that the importanceScores array should be multiplied by.
    If you just want the continuous Jaccard metric, set this to 1.0.
    """
    #                 sum_i (v₁_i ∩ v₂_i)
    #    J(v₂, v₂) = ---------------------
    #                 sum_i (v₁_i ∪ v₂_i)
    #    x ∩ y = min(|x|, |y|) * sign(x) * sign(y)
    #    x ∪ y = max(|x|, |y|)
    return lj.jaccardRegion(importanceScores, scaleFactor, cwm)
# Copyright 2022-2025 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
