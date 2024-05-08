These are not programs you can call, they are python modules. Once BPReveal is installed,
you can import them in your code, like::

    from bpreveal import utils
    utils.easyPredict(mySequence, modelFileName)


Much of the BPReveal API is dedicated to supporting the CLI tools and a typical user won't
need to interact with it. But there are a few functions here and there that you might
find helpful. Here are a few you should know about.


Data processing
---------------

To tile the genome with regions, you can use
:py:func:`bedUtils.makeWhitelistSegments<bpreveal.bedUtils.makeWhitelistSegments>` and
:py:func:`bedUtils.tileSegments<bpreveal.bedUtils.tileSegments>`, or you can use
:py:func:`bedUtils.createTilingRegions<bpreveal.bedUtils.createTilingRegions>`, which
just wraps the two former functions.

For bed intervals, you can resize them with
:py:func:`bedUtils.resize<bpreveal.bedUtils.resize>`.

For working with bigwigs, you can use
:py:func:`utils.loadChromSizes<bpreveal.utils.loadChromSizes>`,
:py:func:`utils.blankChromosomeArrays<bpreveal.utils.blankChromosomeArrays>`, and
:py:func:`utils.writeBigwig<bpreveal.utils.writeBigwig>` to easily write
data to a new bigwig file.

You can use
:py:func:`bedUtils.metapeak<bpreveal.bedUtils.metapeak>` to get the average
profile over many regions, which is useful for plotting.

Making predictions
------------------

If you want to do this the easy way, use the Easy function,
:py:func:`utils.easyPredict<bpreveal.utils.easyPredict>`.
This function will load up a model, make predictions, and then give you the
profiles. It also cleans up after itself and releases the GPU.

For more intense predictions, or if you need the raw model outputs, use
:py:class:`utils.ThreadedBatchPredictor<bpreveal.utils.ThreadedBatchPredictor>`.
This spawns background threads that can run predictions at blinding speed, with
multiple processes sharing the GPU for maximum throughput.
This class supports streaming data, so you can make terabytes of predictions and
save them to disk as they come, letting your program run with a minimal memory
footprint.

If you have model outputs (logits and logcounts) and want a predicted profile, use
:py:func:`utils.logitsToProfile<bpreveal.utils.logitsToProfile>`.

To efficiently convert DNA sequences to and from one-hot-encoded form, use
:py:func:`utils.oneHotEncode<bpreveal.utils.oneHotEncode>` and
:py:func:`utils.oneHotDecode<bpreveal.utils.oneHotDecode>`.
These functions are optimized and can perform their calculations far faster than a naive
implementation with dictionary lookups.


Getting importance scores
-------------------------

If the :py:mod:`interpretFlat<bpreveal.interpretFlat>` CLI tool doesn't do what you need,
you can use
:py:func:`utils.easyInterpretFlat<bpreveal.utils.easyInterpretFlat>` to get
importance scores.
If you need something even more custom, you'll have to wade through the arcane and
complex :py:mod:`interpretUtils<bpreveal.interpretUtils>` module and I'm sorry for you.

Working with motifs
-------------------

The :py:mod:`motifUtils<bpreveal.motifUtils>` module contains helpers for working with
Modisco pattern objects. Typically, you create a
:py:class:`motifUtils.Pattern<bpreveal.motifUtils.Pattern>` object and then call
:py:func:`loadCwm<bpreveal.motifUtils.Pattern.loadCwm>` and then
:py:func:`loadSeqlets<bpreveal.motifUtils.Pattern.loadSeqlets>` to load in the
relevant data.
Just about the only time you'd need to create a Pattern object is to plot it.

Module list
-----------

..
    Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa  # pylint: disable=line-too-long
