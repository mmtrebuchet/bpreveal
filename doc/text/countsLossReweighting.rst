Adaptive Counts Loss
====================

In old BPNet, you had to specify a counts loss weight that was used during
training.
This parameter adjusts how much of the loss is due to the counts component
(mean-squared-error of logcounts) and how much is due to the profile
(multinomial log likelihood).
This is great, but you don't know beforehand what exact number you should apply
to the counts loss in order to get your desired fraction.
As an equation,

.. math::
   f \equiv \frac{\lambda c}{\lambda c + p}


Where :math:`f` is the fraction of loss due to counts, :math:`c` is the raw counts loss,
:math:`p` is the raw profile loss, and :math:`{\lambda}` is the ``counts-loss-weight``
parameter.
If you want, say, ten percent of your loss to come from counts, you need to pick
a value of :math:`{\lambda}` so that that :math:`f = 0.1`.
But since we don't know the values of :math:`c` and :math:`p` that we'll get during
training, we have to guess.

The adaptive counts loss algorithm skirts this issue by updating :math:`{\lambda}`
during training to match your desired :math:`f`.
Starting with the value of ``counts-loss-weight`` you specify in the
configuration file, the algorithm springs into action at the end of each epoch.

.. math::

    \lambda^\prime_{E+1} &= \frac{p f_{target}}{(1-f_{target}) c} \\
    \lambda^{\prime\prime}_{E+1} &= \beta \lambda^\prime_{E+1} + (1 - \beta) \lambda_{E} \\
    \gamma &\equiv \frac{\lambda^{\prime\prime}_{E+1}}{\lambda_E} \\
    \lambda_{E+1} &=
    \begin{cases}
        2 \lambda_E & \gamma > 2 \\
        \lambda^{\prime\prime}_{E+1} & \frac{1}{2} \le \gamma \le 2 \\
        \frac{\lambda_E}{2} & \gamma < \frac{1}{2}
    \end{cases}

where :math:`\lambda_{E+1}` is the :math:`{\lambda}` value for the next epoch,
:math:`\lambda_E`
is the current :math:`\lambda`, :math:`\lambda^{\prime}` is the value of :math:`\lambda` which,
given the current profile loss :math:`p` and counts loss :math:`c`, would give the fraction
target :math:`f_{target}`.
(:math:`f_{target}` should be between 0 (only care about profile) and 1 (only care
about counts), exclusive.
A normal value would be something like 0.1.)
:math:`\lambda^{\prime\prime}` is a smoothed version of :math:`\lambda`, with aggression
parameter :math:`\beta`. :math:`\beta=0` implies no change ever to :math:`\lambda`, while
:math:`\beta = 1` means that the :math:`\lambda` from the last epoch is ignored.
The scripts in bpreveal use a :math:`\beta` parameter of 0.3. This is currently not
user-configurable, but can be edited in :py:mod:`callbacks<bpreveal.callbacks>`.

Early in training, :math:`\lambda_E` may be *way* off from
:math:`\lambda^{\prime\prime}_{E+1}`, and so we clamp :math:`\lambda_{E+1}` to be at most
a factor of two off from :math:`\lambda_E`.

..
    Copyright 2022-2025 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.
