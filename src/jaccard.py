"""Thin wrapper for the internal libjaccard C library."""
import bpreveal.internal.libjaccard as lj


def slidingJaccard(importanceScores, cwm):
    """Calculate the sliding Jaccard similarity.

    :param importanceScores: An array of shape (M, 4) giving hypothetical importance scores.
    :param cwm: An array of shape (N, 4), giving a motif's CWM.
    :return: An array of shape (M - N + 1), giving the sliding Jaccard similarities.

    The returned array is defined by::
        slidingJaccard(A,B)[i] = jaccardRegion(A[i:i+N],B)
    """
    return lj.slidingJaccard(importanceScores, cwm)


def jaccardRegion(importanceScores, scaleFactor, cwm):
    """For given region's importance scores, calculate the continuous Jaccard similarity.

    :param importanceScores: An array of shape (length, 4)
        giving a region's hypothetical importance scores.
    :param scaleFactor: A constant that the importance scores should be multiplied by.
    :param cwm: An array of shape (length, 4) giving the CWM for a motif.
    :return: A single float giving the Jaccard match

    This implements the formula in the modisco paper, namely that::

                     sum_i (v₁_i ∩ v₂_i)
        J(v₂, v₂) = ---------------------
                     sum_i (v₁_i ∪ v₂_i)

    where::

        x ∩ y = min(|x|, |y|) * sign(x) * sign(y)
        x ∪ y = max(|x|, |y|)

    The scaleFactor is a number that the importanceScores array should be multiplied by.
    If you just want the continuous Jaccard metric, set this to 1.0.
    """
    return lj.jaccardRegion(importanceScores, scaleFactor, cwm)
