#include <math.h>
#include <stdio.h>

/* C implementation for sliding Jaccard similarity, based on Ziga Avsec's implementation.
 * There are two functions here that are exported as Python functions through Numpy's amazing
 * f2py module. In order to compile this module, make sure you have gcc and gfortran on your path
 * and then run f2py jaccard.pyf jaccard.c -c
 * and the numpy f2py module will magically create a python module that you can import.
 *
 */

double jaccardIntersection(double x, double y){
    /* Implements
    *     x ∩ y = min(|x|, |y|) * sign(x) * sign(y)
    *  Not exported.
    */
    double minVal = fmin(fabs(x),fabs(y));
    int sameSign = (signbit(x) == signbit(y));
    if(sameSign){
        return minVal;
    }else{
        return -minVal;
    }
}

void jaccardRegion(double *importanceScores, double scaleFactor,
                    double *cwm, int length, double *jaccardValue){
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
    * (length, 4). Note that this function is perfectly fine if importanceScores is longer
    * than length, since it will only look at the first length values in it. For this reason,
    * it is critical that the length is the first dimension, so that we can index into the array
    * without knowing its actual size.
    * the scaleFactor is a number that the importanceScores array should be multiplied by.
    * If you just want the continuous jaccard metric, set this to 1.0.
    *
    * cwm is an array of shape (cwmLength, 4) and represents the CWM that will be scanned.
    *
    * Stores the calculated jaccard distance in jaccardValue.
    *
    */
    double numerator = 0;
    double denominator = 0;
    for(int cwmPos = 0; cwmPos < length; cwmPos++){
        for(int base = 0; base < 4; base++){
            double x = importanceScores[cwmPos*4 + base] * scaleFactor;
            double y = cwm[cwmPos * 4 + base];
            numerator += jaccardIntersection(x,y);
            denominator += fmax(fabs(x), fabs(y));
        }
    }
    *jaccardValue = numerator / denominator;

}

void sumRegion(double *values, int length, double *out){
    /*
     * For an array values of shape (length, 4), calculates the sum of
     * the absolute values of that array. Stores the result in out.
     */
    double ret = 0;
    for (int i = 0; i < length; i++){
         for (int j = 0; j < 4; j++){
            ret += fabs(values[i*4+j]);
         }
    }
    *out = ret;
}


void slidingJaccard(double *importanceScores, int importanceLength,
                    double *cwm, int cwmLength, double *jaccardOut, double *sumsOut){
    /*
     * The meat of the implementation. Given an array importanceScores of shape
     * (importanceLength, 4) and a cwm of shape (cwmLength, 4), calculate the
     * sliding Jaccard similarity as cwm is passed over importanceScores.
     * The arrays jaccardOut and sumsOut both have shape (importanceLength - cwmLength + 1,)
     * because the similarity scores are only calculated where both the CWM and
     * importance score arrays are fully overlapping.
     *
     *
    */
    double jaccardValue;
    double sumOfImportances;
    double sumOfCwm;
    sumRegion(cwm, cwmLength, &sumOfCwm);

    for(int i = 0; i < importanceLength - cwmLength + 1; i++){
        sumRegion(importanceScores+i*4, cwmLength, &sumOfImportances);
        double scaleFactor = sumOfCwm / (sumOfImportances == 0 ? 0.0000001 : sumOfImportances);
        jaccardRegion(importanceScores+i*4, scaleFactor, cwm, cwmLength,
                      &jaccardValue);


        jaccardOut[i] = jaccardValue;
        sumsOut[i] = sumOfImportances;
    }
}


void mysum(double *a, double *out, int rows, int cols){
    for (int i = 0; i < rows; i++){
        out[i] = 0.0;
        for (int j = 0; j < cols; j++){
            out[i] += a[i*cols + j];
        }
    }
}



