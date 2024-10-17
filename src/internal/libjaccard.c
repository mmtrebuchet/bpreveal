#include <math.h>
#define FLOAT_T float

/* C implementation for sliding Jaccard similarity, based on Ziga Avsec's implementation.
 * There are two functions here that are exported as Python functions through Numpy's amazing
 * f2py module. In order to compile this module, make sure you have gcc and gfortran on your path
 * and then run f2py jaccard.pyf jaccard.c -c
 * and the numpy f2py module will magically create a python module that you can import.
 *
 * While this code is entirely my own, the arithmetic is based on the original
 * BPNet code, which is licensed under an MIT-style license. You can find a copy
 * at ``etc/bpnet_license.txt``.
 *
 */


FLOAT_T jaccardIntersection(FLOAT_T x, FLOAT_T y) {
    /* Implements
    *     x ∩ y = min(|x|, |y|) * sign(x) * sign(y)
    *  Not exported.
    */
    FLOAT_T minVal = fmin(fabs(x), fabs(y));
    int sameSign = (signbit(x) == signbit(y));
    if (sameSign) {
        return minVal;
    } else {
        return -minVal;
    }
}

void jaccardRegion(const FLOAT_T * const restrict importanceScores,
                   FLOAT_T scaleFactor, const FLOAT_T * const restrict cwm,
                   int length, int numBases, FLOAT_T *restrict jaccardValue) {
    /*For a particular region of the importance scores, calculate the continuous
    * jaccard similarity. Returns a single number that corresponds to the formula in
    * the modisco paper, namely that
    *
    *               sum_i (v₁_i ∩ v₂_i)
    *   J(v₂, v₂) = --------------------
    *               sum_i (v₁_i ∪ v₂_i)
    *
    * where
    *     x ∩ y = min(|x|, |y|) * sign(x) * sign(y)
    *     x ∪ y = max(|x|, |y|)
    *
    * The importanceScores should be an array of shape
    * (length, numBases). Note that this function is perfectly fine if importanceScores is longer
    * than length, since it will only look at the first length values in it. For this reason,
    * it is critical that the length is the first dimension, so that we can index into the array
    * without knowing its actual size.
    * the scaleFactor is a number that the importanceScores array should be multiplied by.
    * If you just want the continuous jaccard metric, set this to 1.0.
    *
    * cwm is an array of shape (cwmLength, numBases) and represents the CWM that will be scanned.
    *
    * Stores the calculated jaccard distance in jaccardValue.
    *
    */
    FLOAT_T numerator = 0;
    FLOAT_T denominator = 0;
    for (int cwmPos = 0; cwmPos < length; cwmPos++) {
        for (int base = 0; base < numBases; base++) {
            FLOAT_T x = importanceScores[cwmPos*numBases + base] * scaleFactor;
            FLOAT_T y = cwm[cwmPos * numBases + base];
            numerator += jaccardIntersection(x, y);
            denominator += fmax(fabs(x), fabs(y));
        }
    }
    *jaccardValue = numerator / denominator;
}

void sumRegion(const FLOAT_T * const restrict values,
               int length, int numBases, FLOAT_T *restrict out) {
    /*
     * For an array values of shape (length, numBases), calculates the sum of
     * the absolute values of that array. Stores the result in out.
     */
    FLOAT_T ret = 0;
    for (int i = 0; i < length; i++) {
         for (int j = 0; j < numBases; j++) {
            ret += fabs(values[i*numBases+j]);
         }
    }
    *out = ret;
}


void slidingJaccard(const FLOAT_T * const restrict importanceScores,
                    int importanceLength, const FLOAT_T * const restrict cwm,
                    int cwmLength, int numBases, FLOAT_T *restrict jaccardOut,
                    FLOAT_T *restrict sumsOut) {
    /*
     * The meat of the implementation. Given an array importanceScores of shape
     * (importanceLength, numBases) and a cwm of shape (cwmLength, numBases), calculate the
     * sliding Jaccard similarity as cwm is passed over importanceScores.
     * The arrays jaccardOut and sumsOut both have shape (importanceLength - cwmLength + 1,)
     * because the similarity scores are only calculated where both the CWM and
     * importance score arrays are fully overlapping.
     *
     *
    */
    FLOAT_T jaccardValue;
    FLOAT_T sumOfImportances;
    FLOAT_T sumOfCwm;
    sumRegion(cwm, cwmLength, numBases, &sumOfCwm);

    for (int i = 0; i < importanceLength - cwmLength + 1; i++) {
        sumRegion(importanceScores+i*numBases, cwmLength, numBases,
                  &sumOfImportances);
        FLOAT_T scaleFactor = sumOfCwm /
            (sumOfImportances == 0 ? 0.0000001 : sumOfImportances);
        jaccardRegion(importanceScores+i*numBases, scaleFactor, cwm,
                      cwmLength, numBases, &jaccardValue);

        jaccardOut[i] = jaccardValue;
        sumsOut[i] = sumOfImportances;
    }
}

/*Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.*/
