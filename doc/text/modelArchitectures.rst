Model architectures
===================

The precise details of the model architectures can be found in models.py, but
they share some common themes. Every model that ever gets saved to disk accepts
a one-hot encoded sequence as input, and produces outputs that are grouped into
"heads". A model may generate any number of heads, and the heads may have
different sizes. In general, each head should represent one set of DNA
fragments. For example, an experiment that produces cut sites on the + and -
strand of DNA produces two tracks, but the tracks represent two ends of the
same fragments. So these two tracks would be in the same head. However, if you
have an experiment where it's appropriate to split fragments into "short"
(100-500 bp) and "long" (1 kb to 10 kb), then those tracks do not represent the
same fragments, so they should be in different heads.

If you have done ChIP-nexus on three different factors, then you'd have three
heads, each one corresponding to a different factor, and each head would
predict both the + and - strand data for that factor.

If you're not sure if you can combine your data under one output head, it's
much safer to split the data into multiple heads.

A head contains a profile prediction and a counts prediction. The profile
prediction is a tensor of shape ``(batch-size x) number-of-tracks x
output-width``, and each value in this tensor is a logit. Note that the *whole*
profile prediction should be considered when taking the softmax. That is to
say, the profile of the first track is NOT
:math:`e^{logcounts} * softmax(profile_{0,:})`, but rather you have to take the
softmax first and then slice out the profile:
:math:`e^{logcounts} * softmax(profile)_{0,:}`. There is a function,
:py:func:`logitsToProfile<bpreveal.utils.logitsToProfile>`, that does this
automatically.

Of course, if the profile only has one track, this distinction is vacuous.
The counts output is a scalar that represents the natural logarithm of the
number of reads predicted for the current region.

It is possible to add more model architectures, but currently the program only
supports a BPNet-style architecture. You can take a look at soloModel in
:py:mod:`models<bpreveal.models>` for details on how it works.

..
    Copyright 2022-2025 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.
