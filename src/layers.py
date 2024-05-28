"""Custom layers that are needed for the various models."""
from bpreveal import logUtils
import tensorflow as tf
import tf_keras as keras
from tensorflow.math import add, log, exp, reduce_min, negative, multiply
from tensorflow.nn import elu

class LinearRegression(keras.layers.Layer):
    """A simple layer that performs a linear regression on its inputs.

    .. highlight:: none

    Implements the following formula::

        out = input * slope + offset

    where slope and offset are two (scalar) parameters.

    :param kwargs: Passed to the keras Layer initializer.

    This class correctly handles shap attribution, and should be used for all
    transformation models.
    """

    def __init__(self, **kwargs):
        """Construct."""
        super().__init__(**kwargs)
        self.slope = self.add_weight(
            shape=(1,),
            initializer="Ones",
            dtype="float32",
            trainable=True,
            name='slope')

        self.offset = self.add_weight(
            initializer="Zeros",
            shape=(1,),
            dtype='float32',
            trainable=True,
            name='offset')

        self.constantOne = self.add_weight(
            initializer="Ones",
            shape=(1,),
            dtype="float32",
            trainable=False,
            name="C")

        self.constantK = self.add_weight(
            initializer="Ones",
            shape=(1,),
            dtype="float32",
            trainable=False,
            name="C")
        self.constantK.assign((3,))

        self.xOffset = self.add_weight(
            initializer="Ones",
            shape=(1,),
            dtype="float32",
            trainable=False,
            name="D")

        self.scaledInp = self.add_weight(
            initializer="Ones",
            shape=(1,),
            dtype="float32",
            trainable=False,
            name='SI'
        )

        self.logSlope = self.add_weight(
            initializer="Ones",
            shape=(1,),
            dtype="float32",
            trainable=False,
            name='LS'
        )

        self.slopeSum = self.add_weight(
            initializer="Ones",
            shape=(1,),
            dtype="float32",
            trainable=False,
            name='SS'
        )

        self.minRet = self.add_weight(
            initializer="Ones",
            shape=(1,),
            dtype="float32",
            trainable=False,
            name='MR'
        )

        self.inputLog = self.add_weight(
            initializer="Ones",
            shape=(1,),
            dtype="float32",
            trainable=False,
            name='IL'
        )


    def call(self, inputs, *args, **kwargs):
        """Actually perform the calculation.

        Variadic arguments are ignored.
        """
        # y = m x + b
        #   = m * x + b + md - md
        #   = m * x + b + md - md + (x + d)k - (x + d)k
        #   let x' = x + d
        #   = m * x + b + md - md + x'k - x'k
        #   = m * x + m * d + b - md + x'k - x'k
        #   = mx' + b - md + x'k - x'k
        #   = (m + k)x' + b - md - x'k
        #   Let m' = m + k
        #   = m'x' + b - md - x'k
        #   = exp(log(m'x')) + b - md - x'k
        #   = exp(log(m') + log(x')) + b - md - x'k
        #   m' approx (elu(m' - 1) + 1)
        #   m'' = elu(m' - 1) + 1
        # y = exp(log(m'') + log(x')) + b - md - x'k
        #   let a = log(m'') + log(x')
        #   = exp(a) + b - md - x'k
        #   let v = - md - x'k
        #   = exp(a) + b + v
        # let d = 1 - min(x)
        # let k=3
        x = inputs
        m = self.slope
        b = self.offset
        minelem = reduce_min(x)
        rawOffset = negative(minelem)
        d = add(rawOffset, self.constantOne)
        k = self.constantK
        xPrime = add(x, d)
        mPrime = add(m, k)
        mDoublePrime = add(elu(add(mPrime, negative(self.constantOne))), self.constantOne)
        a = add(log(mDoublePrime), log(xPrime))
        md = multiply(m, d)
        xPrimeK = xPrime
        for i in range(3 - 1):
            xPrimeK = add(xPrimeK, xPrime)
        negMd = negative(md)
        negXPrimeK = negative(xPrimeK)
        v = add(negMd, negXPrimeK)
        rhs = add(v, b)

        ret = add(exp(a), rhs)
        return ret
        # This is real dumb.
        # So the deepshap code blows up if I use a straight multiplication here.
        # After a few hours of fruitless debugging, I realized that it's okay with
        # addition, exponentiation, and taking logs.
        # So...
        # y = m * x + b
        #   = m * x + b + m * d - m * d
        #   = m * x + b + m * d - m * d + (x + d) * k - (x + d) * k
        #   = m * (x + d) + b - m * d + (x + d) * k - (x + d) * k
        #   = (m + k) * (x + d) + b - m * d - (x + d) * k
        #   = exp(log((m + k) * (x + d)) + b - m * d - (x + d) * k
        #   = exp(log(m + k) + log(x + d)) + b - m * d - (x + d) * k
        #
        # y = m x + b
        # y = (m + k) x + b - x k
        # where d is large enough so that (x + d) > 0 ∀ x
        # Therefore I'll use d = 1 - min(x)
        # and k is an integer (this is important!) that is large enough so that
        # m + k > 0.
        # To ensure that m + k > 0, I'll set k = 3 and
        # m' = relu(m + 3) + 3

        # shiftSlope = relu(add(self.slope, self.constantK))
        # useSlope = add(shiftSlope, self.constantK)
        # minelem = reduce_min(inputs)
        # rawOffset = negative(minelem)
        # offset = add(rawOffset, self.constantOne)  # This is d.
        # # self.xOffset.assign(offset)
        # scaledInputs = add(x=inputs, y=offset)  # x + d
        # # minScaled = reduce_min(scaledInputs)
        # # self.scaledInp.assign(add(minScaled, self.constantOne))
        # inputLog = log(x=scaledInputs)
        # # self.inputLog.assign(add(reduce_min(inputLog), self.constantOne))
        # logslope = log(x=useSlope)
        # # self.logSlope.assign(add(reduce_min(logslope), self.constantOne))
        # slopeSum = add(x=inputLog, y=logslope)
        # # self.slopeSum.assign(add(reduce_min(slopeSum), self.constantOne))
        # mx = exp(x=slopeSum)
        #
        # negativeOffset = negative(offset)
        # negativeMd = multiply(x=self.slope, y=negativeOffset)
        # mxMinusMd = add(mx, negativeMd)
        # firstRet = add(x=mxMinusMd, y=self.offset)
        # # self.minRet.assign(add(reduce_min(firstRet), self.constantOne))
        # tripleScaledInputs = add(scaledInputs, add(scaledInputs, scaledInputs))
        # secondRet = add(firstRet, negative(tripleScaledInputs))
        # return secondRet


class LinearRegression2(keras.layers.Layer):
    """A simple layer that performs a linear regression on its inputs.

    .. highlight:: none

    Implements the following formula::

        out = input * slope + offset

    where slope and offset are two (scalar) parameters.

    :param kwargs: Passed to the keras Layer initializer.

    This class correctly handles shap attribution, and should be used for all
    transformation models.
    """

    def __init__(self, **kwargs):
        """Construct."""
        super().__init__(**kwargs)
        self.slope = self.add_weight(
            shape=(1,),
            initializer="Ones",
            dtype="float32",
            trainable=True,
            name='slope')

        self.offset = self.add_weight(
            initializer="Zeros",
            shape=(1,),
            dtype='float32',
            trainable=True,
            name='offset')

        self.constantOne = self.add_weight(
            initializer="Ones",
            shape=(1,),
            dtype="float32",
            trainable=False,
            name="C")

        self.xOffset = self.add_weight(
            initializer="Ones",
            shape=(1,),
            dtype="float32",
            trainable=False,
            name="D")

    def call(self, inputs, *args, **kwargs):
        """Actually perform the calculation.

        Variadic arguments are ignored.
        """
        # This is real dumb.
        # So the deepshap code blows up if I use a straight multiplication here.
        # After a few hours of fruitless debugging, I realized that it's okay with
        # addition, exponentiation, and taking logs.
        # So...
        # y = m * x + b
        #   = m * x + b + m * d - m * d
        #   = m * (x + d) + b - m * d
        #   = exp(log(m * (x + d)) + b - m * d
        #   = exp(log(m) + log(x + d)) + b - m * d

        # where d is large enough so that (x + d) > 0 ∀ x
        # Therefore I'll use d = 1 - min(x)

        minelem = reduce_min(inputs)
        rawOffset = negative(minelem)
        offset = add(rawOffset, self.constantOne)  # This is d.
        self.xOffset.assign(offset)
        scaledInputs = add(x=inputs, y=offset)  # x + d

        inputLog = log(x=scaledInputs)
        logslope = log(x=self.slope)
        slopeSum = add(x=inputLog, y=logslope)
        mx = exp(x=slopeSum)

        negativeOffset = negative(offset)
        negativeMd = multiply(x=self.slope, y=negativeOffset)
        mxMinusMd = add(mx, negativeMd)
        return add(x=mxMinusMd, y=self.offset)
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
