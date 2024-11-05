#define FLOAT_T float
#define SIZE_T long
/*
 * C implementation of row sliding, used to prepare input data for training.
 *
 */

void runRow(const FLOAT_T * const restrict source, FLOAT_T *restrict dest,
            SIZE_T numRows, SIZE_T numSourceCols, SIZE_T numDestCols, SIZE_T depth,
            const int * const restrict rowIndexes,
            const int * const restrict colIndexes,
            SIZE_T row) {
    for (SIZE_T destCol = 0; destCol < numDestCols; destCol++) {
        for (SIZE_T z = 0; z < depth; z++) {
            SIZE_T sr = rowIndexes[row];
            FLOAT_T srcVal = source[row * numSourceCols * depth
                                    + (destCol + colIndexes[row]) * depth
                                    + z];
            dest[sr * numDestCols * depth
                + destCol * depth
                + z] = srcVal;
        }
    }
}

void runRowChar(const unsigned char * const restrict source,
                FLOAT_T *restrict dest, SIZE_T numRows, SIZE_T numSourceCols,
                SIZE_T numDestCols, SIZE_T depth,
                const int * const restrict rowIndexes,
                const int * const restrict colIndexes, SIZE_T row) {
    for (SIZE_T destCol = 0; destCol < numDestCols; destCol++) {
        for (SIZE_T z = 0; z < depth; z++) {
            SIZE_T sr = rowIndexes[row];
            FLOAT_T srcVal = source[row * numSourceCols * depth
                                    + (destCol + colIndexes[row]) * depth
                                    + z];
            dest[sr * numDestCols * depth
                + destCol * depth
                + z] = srcVal;
        }
    }
}

void slide(const FLOAT_T * const restrict source, FLOAT_T *restrict dest,
           int numRows, int numSourceCols, int numDestCols, int depth,
           const int * const restrict rowIndexes,
           const int * const restrict colIndexes) {
    /**
    * source is a (numRows x numSourceCols x depth) array.
    * dest is a (numRows x numDestCols x depth) array.
    * rowIndexes is a (numRows,) vector.
    * colIndexes is a (numRows,) vector.
    * Implements the following:
    * for row in numRows:
    *   dest[rowIndexes[row]] = source[row, colIndexes[row]:colIndexes[row]+numDestCols]
    *
    * This is done on all the data in every epoch, so speeding it up made sense.
    * It is invalid for any of the arrays to overlap at all.
    *
    */
    #pragma omp parallel for num_threads(8)
    for (SIZE_T row = 0; row < numRows; row++) {
        runRow(source, dest, numRows, numSourceCols, numDestCols,
               depth, rowIndexes, colIndexes, row);
    }
}

void slideChar(const unsigned char * const restrict source,
               FLOAT_T *restrict dest, int numRows, int numSourceCols,
               int numDestCols, int depth,
               const int * const restrict rowIndexes,
               const int * const restrict colIndexes) {
    /**
    * source is a (numRows x numSourceCols x depth) array.
    * dest is a (numRows x numDestCols x depth) array.
    * rowIndexes is a (numRows,) vector.
    * colIndexes is a (numRows,) vector.
    * Implements the following:
    * for row in numRows:
    *   dest[rowIndexes[row]] = source[row, colIndexes[row]:colIndexes[row]+numDestCols]
    *
    * This is done on all the data in every epoch, so speeding it up made sense.
    * It is invalid for any of the arrays to overlap at all.
    *
    */
    #pragma omp parallel for num_threads(8)
    for (SIZE_T row = 0; row < numRows; row++) {
        runRowChar(source, dest, numRows, numSourceCols, numDestCols,
                   depth, rowIndexes, colIndexes, row);
    }
}
/*Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.*/
