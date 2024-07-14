
Breaking Changes
================

A more detailed version history can be found in the :doc:`changelog`.

BPReveal 6.x
------------

BPReveal 6.0.0
^^^^^^^^^^^^^^

When BPReveal 6.0.0 is released, the following breaking changes will occur:

1. The :py:mod:`showModel<bpreveal.showModel>` tool will be removed since it just
   wraps a single keras function call, and getting graphical output requires two
   rather large dependencies.
2. ``interpretPisaFasta`` and ``interpretPisaBed`` are old names for
   :py:mod:`interpretPisa<bpreveal.interpretPisa>`. They exist only as symlinks
   in the bin/ directory, and using them has emitted a warning since 4.0.0.
   In BPReveal 6.0.0, the symlinks will be removed.
3. ``makePredictionsFasta`` and ``makePredictionsBed`` are old names for makePredictions.
   They exist only as symlinks in the bin/ directory, and using them has emitted a
   warning since 4.1.1. The old names will be removed.
4. Not including a ``correct-receptive-field`` field in your config to interpretPisa will
   no longer issue a warning - the correct receptive field will be applied by default.
5. If you want to get a tsv of seqlets from
   :py:mod:`motifSeqletCutoffs<bpreveal.motifSeqletCutoffs>`, you will need to provide a
   ``modisco-window`` parameter in the configuration json. Omitting this has issued
   a warning and generated invalid coordinate data since 4.1.3.
6. The ``plots.py`` module in the tools directory will be removed. Its replacement,
   ``plotting.py``, was introduced in 4.2.0 and is part of the main repository.
7. The ``addNoise`` tool will be removed. It turns out that it's not useful.
8. When doing PISA interpretation on a fasta, you must call the fasta
   ``fasta-file`` instead of ``sequence-fasta``. (Has emitted a warning
   since 4.0.0.)
9. :py:mod:`prepareBed<bpreveal.prepareBed>` will no longer accept a list of
   bigwigs without head information. The program currently spits out warnings
   when you do this, and these will become an error.
10. Using an importance hdf5 from before version 4.0.0 will now result in an
    error instead of a deprecation warning.


BPReveal 5.x
------------

BPReveal 5.0.0
^^^^^^^^^^^^^^

When BPReveal 5.0.0 was released, the following breaking changes occurred:

1. The first argument to
   :py:func:`models.transformationModel<bpreveal.models.transformationModel>`
   was renamed to get rid of a name collision that pylint gets upset about.
2. The ``correct-receptive-field`` flag in :py:mod:`interpretPisa<bpreveal.interpretPisa>`,
   introduced in 4.1.2, switched from being ``false`` by default to being ``true``
   by default. This fixes an off-by-one bug in how receptive field was calculated.
3. The ``dumpModiscoSeqlets`` tool was removed, since it's not useful.
4. BPReveal now uses Tensorflow 2.16 and Keras 3.0. This will cause some
   breaking changes. Models are saved on disk now using a ``.keras`` extension
   because Keras 3.0 enforces this.
5. Keras 3.0 only reports a whole-model loss instead of a per-output loss, which
   caused the names of the reported metrics to change. Instead of
   ``solo_logcounts_nanog_loss`` and ``solo_profile_nanog_loss``, these are now
   ``solo_logcounts_nanog_reweightable_mse`` and ``solo_profile_nanog_multinomial_nll``.
   Technically, these are now *metrics* and not *losses*, but that should make no
   difference in practice.
6. The shap code was further trimmed down. The names of the arguments to
   ``combine_mult_and_diffref`` were changed to camelCase to match the style of BPReveal.

BPReveal 4.x
------------

BPReveal 4.3.0
^^^^^^^^^^^^^^
1. Some of the arguments in :py:mod:`internal.plotUtils<bpreveal.internal.plotUtils>`
   were renamed to improve consistency.
2. The internal implementation of transformation models was changed so that they can
   be interpreted with shap. If you were messing with the internal layers in a
   transformation model, they're different now. If you're not probing at the internal
   layers, this will have no effect - the API is unchanged.

BPReveal 4.2.0
^^^^^^^^^^^^^^
1. BPReveal now uses tensorflow 2.16 and Python 3.12. It still uses the legacy
   Keras, though. If you were manually working with Keras, you will need to
   import ``tf_keras`` instead.
2. The tools.plots module has been retired. It has been replaced by
   :py:mod:`plotting<bpreveal.plotting>`, which exposes a semi-coherent API and has
   generally been cleaned way up. The old module now emits a warning, but it will stay
   around until at least version 6.0.0.
3. The names of the type variables in the :py:mod:`gaOptimise<bpreveal.gaOptimize>` were
   switched to UPPER_CASE to match the rest of the project. This should have no effect
   on user code.

BPReveal 4.1.4
^^^^^^^^^^^^^^
1. The shap code was replaced with the current release from upstream.
   This should not break anything unless you were doing something *really* weird.

BPReveal 4.1.3
^^^^^^^^^^^^^^
1. With the creation of the new :py:class:`Seqlet<bpreveal.motifUtils.Seqlet>` class,
   several arrays that used to be in the :py:class:`Pattern<bpreveal.motifUtils.Pattern>`
   class have been removed. If you were creating Patterns in your own code, you will need
   to instead refer to the seqlet arrays. No file formats are changed by this.


BPReveal 4.1.1
^^^^^^^^^^^^^^
1. The name of the counts head in a transformation model that uses bias counts
   changed from ``combined_log_counts`` to ``combined_logcounts``, which might
   possibly break some very obscure use case. This change was necessary to fix
   a couple bugs with :doc:`adaptive counts loss<countsLossReweighting>` and
   :py:mod:`showTrainingProgress<bpreveal.showTrainingProgress>`.

2. To allow the prediction script to work with very large bed files, some refactoring was
   done. This included moving functions to add metadata to hdf5 files into a new module,
   :py:mod:`internal.predictUtils<bpreveal.internal.predictUtils>`. If you were calling
   them from the old ``makePredictionsFasta.py`` module, they have moved.

3. A new library, libslide, has been added. You will need to re-run make (or reinstall
   the environment) to use it.

BPReveal 4.1.0
^^^^^^^^^^^^^^
The output format from training was totally re-written to be easier to use in log files.
A new tool, :py:mod:`showTrainingProgress<bpreveal.showTrainingProgress>` can be used to
get a nice view of your model's progress as it trains up.


BPReveal 4.0.2
^^^^^^^^^^^^^^

A few variables deep inside BPReveal were changed from snake_case to the
preferred style for BPReveal, camelCase. This should not have any effect unless
you were digging deep inside the code.

BPReveal 4.0.0
^^^^^^^^^^^^^^
The following breaking changes occurred when BPReveal 4.0.0 was released:

1. The chromosome list in the hdf5 files produced by
   :py:mod:`interpretFlat<bpreveal.interpretFlat>` stored chromosome
   information as strings, unlike all other output file formats. This changed
   so that chromosomes are numbered.
   :py:mod:`shapToBigwig<bpreveal.shapToBigwig>` and the motif scanning
   utilities now emit warnings if they detect an old-style importance hdf5.
   This will become an error in 5.0.0.
2. The adaptive loss algorithm required me to implement a custom mse loss. In
   3.6, I sneakily called it ``"mse"`` so you didn't have to add another custom
   object to scope when you load a new model. This loss will be renamed
   ``"reweightableMse"`` and you'll have to add it to the custom object scopes
   when you load a model. Since the full new loss includes a tensor that must
   be created beforehand, :py:mod:`losses<bpreveal.losses>` will include a
   dummy version that you can use to load, but not train, a model. See
   :doc:`countsLossReweighting` for the algorithm.

BPReveal 3.x
------------

BPReveal 3.6.0
^^^^^^^^^^^^^^

1. The :py:mod:`predictToBigwig<bpreveal.predictToBigwig>` script now averages the values in
   overlapping regions instead of taking the leftmost base.
   This may result in small changes in generated bigwigs.
2. In order to accommodate the adaptive loss algorithm
   (:doc:`countsLossReweighting`), some of the layer names in transformation
   models were changed. If you were depending on these layer names, I'm curious
   to know how you got yourself in that situation.

BPReveal 3.5.0
^^^^^^^^^^^^^^

1. BPReveal now uses Python 3.11, instead of 3.10. Users must re-build
   the ``libjaccard`` library for the new Python version.

BPReveal 3.0.0
^^^^^^^^^^^^^^
1. You must specify a ``"remove-overlaps"`` field in configuration files for
   :py:mod:`prepareBed<bpreveal.prepareBed>`.
2. ``cropdown`` layers were removed as an option for transformation models.
3. The transformation model configuration file calls the input length
   ``input-length`` instead of ``sequence-input-length``.

..
    Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.
