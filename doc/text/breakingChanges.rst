
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
   They exist only as symlinks in the bin/ directory, and using them has emitted a warning
   since 4.1.1. The old names will be removed.


BPReveal 5.x
------------

BPReveal 5.0.0
^^^^^^^^^^^^^^
When BPReveal 5.0.0 is released, the following breaking changes will occur:

1. :py:mod:`prepareBed<bpreveal.prepareBed>` will no longer accept a list of
   bigwigs without head information. The program currently spits out warnings
   when you do this, and these will become an error.
2. Using an importance hdf5 from before version 4.0.0 will now result in an
   error instead of a deprecation warning.
3. When doing PISA interpretation on a fasta, you must call the fasta
   ``fasta-file`` instead of ``sequence-fasta``. (Has emitted a warning
   since 4.0.0.)
4. The first argument to
   :py:func:`models.transformationModel<bpreveal.models.transformationModel>`
   will be renamed to get rid of a name collision that pylint gets upset about.

BPReveal 4.x
------------

BPReveal 4.1.1
^^^^^^^^^^^^^^
1. The name of the counts head in a transformation model that uses bias counts changed from
   ``combined_log_counts`` to ``combined_logcounts``, which might possibly break some very
   obscure use case. This change was necessary to fix a couple bugs with
   :doc:`adaptive counts loss<countsLossReweighting>` and
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

