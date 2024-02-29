Changelog
=========

This is the detailed changelog. If you're just looking for breaking changes,
see :doc:`breakingChanges`.


BPReveal 4.x
------------

BPReveal 4.1.x
^^^^^^^^^^^^^^



BPReveal 4.1.1, 2024-02-27
''''''''''''''''''''''''''

NEW FEATURES:
    * The PISA code now runs in parallel if you provide a ``num-threads`` parameter
      in its configuration file. Three-fold speedup is very possible.

ENHANCEMENTS:
    * Integrated documentation from Melanie on the motif scanning tools.
    * Separated type definitions out from the utils module into a new internal.constants
      module so that the utils documentation isn't full of type annotations.
    * Combined the old makePredictionsBed and makePredictionsFasta into a single
      makePredictions script. The old names will be removed in 6.0.0.
    * You can specify a genome name for ``background-probs`` in
      :py:mod:`motifSeqletCutoffs<bpreveal.motifSeqletCutoffs>` and
      :py:mod:`motifScan<bpreveal.motifScan>`.
    * Rewrote the generator to use a new C library, making the data loading step
      at the end of each batch about three times faster. The jitter values will be
      slightly different than before since I'm using the random number generator
      differently, but there should be no problems with backwards compatibility.
      Hooray for better GPU utilization!

BUG FIXES:
    * Fixed the name of the counts head in transformation models using bias counts from
      ``combined_log_counts_<headname>`` to ``combined_logcounts_<headname>``, making
      ``use-bias-counts`` compatible with adaptive loss and the new training progress
      logger. (Melanie Weilert)
    * Corrected a bug where non-links in the documentation still showed up as blue.
      (Thanks to Patrick Moeller for the fix!)

CONTRIBUTORS:
    Melanie Weilert, Patrick Moeller, Charles McAnany

BPReveal 4.1.0, 2024-02-16
''''''''''''''''''''''''''

BREAKING CHANGES:
    * The output from training now has a radically different format. If you were parsing
      progress bars from log files, I hope that the new format will make your life
      easier.

NEW FEATURES:
    * Extracted the logging functions into a new module,
      :py:mod:`logUtils<bpreveal.logUtils>`. It separates BPReveal logging into
      its own class of messages, so you can still use logging with your own
      code without stepping on BPReveal's toes.
    * Removed the old progress bar logging system during training. Training now produces
      a spew of logging messages that are easier to grep, and they can be displayed in
      real time by the new :py:mod:`showTrainingProgress<bpreveal.showTrainingProgress>`
      tool. This tool requires training the model with INFO or DEBUG verbosity, otherwise
      no useful output is produced. The format of the output is still flexible and will
      not be finalized until 4.3.0.

ENHANCEMENTS:
    * Added parallelization to :py:mod:`prepareBed<bpreveal.prepareBed>`.
      It should now be a lot faster. Output is bit-for-bit identical.
    * Dramatically sped up the whitelist calculation for tiling the genome in
      :py:func:`makeWhitelistSegments<bpreveal.bedUtils.makeWhitelistSegments>`.
    * Made the verbosity of the training step match the user-specified verbosity.
      If your configuration json says that verbosity should be ``WARNING``, then there
      is much less output from the training scripts.
    * Switched the documentation to a serif font.
    * Cleaned up the documentation building process a lot.

DEPRECATIONS:
    * The showModel script is deprecated and will be removed in 6.0.0.
      It does very little and required two large dependencies (pydot and graphviz)
      to get the image out.

BUG FIXES:
    * The :py:mod:`motifAddQuantiles<bpreveal.motifAddQuantiles>` script used to add
      a new copy of quantile information if the file already had that data. Now it
      replaces the old quantile information.

CONTRIBUTORS:
    Charles McAnany

BPReveal 4.0.x
^^^^^^^^^^^^^^

Version 4.0.4, 2024-02-07
'''''''''''''''''''''''''

BUG FIXES:
    * Fixed a bug that prevented ``null`` quantile cutoffs in
      :py:mod:`motifAddQuantiles<bpreveal.motifAddQuantiles>`.

CONTRIBUTORS:
    Charles McAnany

Version 4.0.3, 2024-01-30
'''''''''''''''''''''''''

BUG FIXES:
    * Fixed a bug in the Easy prediction function incorrectly assuming that models
      had only one output.
    * Added pydot and graphviz as optional components in the build script, only
      necessary to use the graphical output from showModel.

CONTRIBUTORS:
    Charles McAnany

Version 4.0.2, 2024-01-29
'''''''''''''''''''''''''

BREAKING CHANGES:
    * A few internal variable names were switched from snake_case to camelCase.
      This should not have any effect on code that uses BPReveal.

NEW FEATURES:
    * Added a feature to
      :py:mod:`makePredictionsFasta<bpreveal.makePredictionsFasta>` where you
      can specify a bed file and a genome. If you do, then the coordinate
      information from that bed will be saved in the output h5 and you can use
      :py:mod:`predictToBigwig<bpreveal.predictToBigwig>` with it. Added the
      same feature to interpretFlat, so you can use it with
      :py:mod:`shapToBigwig<bpreveal.shapToBigwig>`.
    * Two new functions:
      :py:func:`utils.blankChromosomeArrays<bpreveal.utils.blankChromosomeArrays>`
      and :py:func:`utils.writeBigwig<bpreveal.utils.writeBigwig>`

ENHANCEMENTS:
    * A complete overhaul of the documentation means that we now have on-line
      docs for all of the components of BPReveal, all with type annotations.
      The old overview.pdf has been removed and split up across many webpages.
    * Many functions that were previously undocumented are now
      fully-documented.
    * Automated the testing of schemas. The runTests.py script will
      automatically go through all the test cases.
    * Added new arguments to
      :py:func:`utils.loadChromSizes<bpreveal.utils.loadChromSizes`. These let
      you pass in things other than a ``chrom.sizes`` file name. You can now
      provide a genome fasta, a bigwig, and a bunch of other things.


BUG FIXES:
    * The dummy progress bar for an int passed to
      :py:func:`utils.wrapTqdm<bpreveal.utils.wrapTqdm>` returned the dummyPbar
      *class*, not an *object*. This has been fixed.

CONTRIBUTORS:
    Charles McAnany

Version 4.0.1, 2024-01-17
'''''''''''''''''''''''''

NEW FEATURES:
    * Added the option to specify the kmer size for the shuffles in shap value
      calculations. interpretFlat and interpretPisa now have an optional
      "kmer-size" parameter in their configuration jsons. If omitted, the
      default (non-kmer-preserving) shuffle is performed.
    * There are now easy functions that you can use to make predictions and get
      interpretation scores in :py:mod:`utils<bpreveal.utils>`.
    * A new
      :py:class:`ThreadedBatchPredictor<bpreveal.utils.ThreadedBatchPredictor>`
      runs predictions in another thread, and lets you hold it in a context
      manager so that it shuts down and starts up when you need it.


ENHANCEMENTS:
    * All BPReveal programs that take JSON input now check that input against a
      schema.
    * Lots of enhancements to the pisa plotting tools!

BUG FIXES:
    * :py:mod:`makePredictionsFasta<bpreveal.makePredictionsFasta>` used a
      non-iterable tqdm object as an iterable in a loop. This has been fixed.

Version 4.0.0, 2024-01-10
'''''''''''''''''''''''''

BREAKING CHANGES:
    * interpretFlat now produces h5 files that use integer indexes for the
      chromosome instead of strings. Internal programs that were affected by
      this change now emit a warning if they detect an importance file from an
      earlier release.
    * The adaptive loss is now named reweightableMse, and comes from a function
      in losses.py called weightedMse. If you're just loading a model, you can
      specify "custom_objects={'multinomialNll': losses.multinomialNll,
      'reweightableMse': losses.dummyMse}" when you call load_model in keras.
      There's also a new loadModel function in utils.py that does this for you.

DEPRECATIONS:
    * interpretPisaBed and interpretPisaFasta have been merged into one
      program, interpretPisa. Symlinks exist in the bin/ directory; using one
      will generate a warning until 6.0.0, when the symlinks will be removed.
    * interpretPisa now expects a property called "fasta-file", (consistent
      with interpretFlat), instead of the old "sequence-fasta" property. This
      will generate a warning until 6.0.0, when it will become an error.
    * The old json format for prepareBed has produced a warning since 3.3.1. It
      will be an error in 5.0.0
    * Using an old-style importance score hdf5 (with string chromosome names)
      is now a warning, and will become an error in 6.0.0.


ENHANCEMENTS:
    * All queues now have a timeout, so that a crash in one thread will
      propagate through the entire program instead of freezing.
    * Started working on json schemas to validate inputs, hopefully making
      errors less opaque. All of the programs except the motif scanners have
      schemas now. (Thanks to Melanie for lighting the fire that led to this!)
    * Plenty of code cleanups and tweaks.


BUG FIXES:
    * Fixed a typo in interpretUtils.py (Thanks, Haining!)
    * The specification incorrectly stated that the warning level of verbosity
      was "WARN", when in fact it should be "WARNING".
    * Fixed the install script to use tensorflow 2.15, which requires cuda 12.
    * The automatic memory allocation in interpretFlat
      (utils.py/limitMemoryUsage) worked incorrectly if running on a MIG gpu.
      This has been remedied with an extremely ugly hack that looks at
      CUDA_VISIBLE_DEVICES and sees if there's a MIG entry. If so, it estimates
      the available memory based on the MIG type's name (like 3g.20gb).

CONTRIBUTORS:
    Charles McAnany, Haining Jiang, Melanie Weilert

BPReveal 3.x
------------

BPReveal 3.6.x
^^^^^^^^^^^^^^

Version 3.6.1, 2023-12-05
'''''''''''''''''''''''''

ENHANCEMENTS:
    * Added a version of ushuffle that is compatible with python 3.11. This is
      now part of the main bpreveal repository, in the src/internal directory.
    * Implemented an adaptive counts loss weight algorithm, so you can specify
      the fraction of the loss due to counts instead of a raw :math:`{\lambda}`
      parameter.

CONTRIBUTORS:
    Charles McAnany


Version 3.6.0, 2023-11-06
'''''''''''''''''''''''''

ENHANCEMENTS:
    * The old predictToBigwig script had odd behavior with overlapping inputs.
      It always took the leftmost region that predicted a particular base and
      saved that out. Now, it instead averages all of the predictions made for
      a given base and saves the average value. This may result in small
      changes to your bigwigs, but should not cause any meaningful differences.
      predictToBigwig now has a --threads option, since I made it GO FAST LIKE
      NYOOOOM! Since this cause a change in outputs, I'm assigning a minor
      version increase, though it's really not a big deal.
    * Fully qualified the names of all the imports in all the python files, so
      they should be callable from anywhere and importable from any script now.

BUG FIXES:
    * In the specification, corrected "chrom_name" to "chrom_names" in the
      output hdf5 format for makePredictionsBed.

CONTRIBUTORS:
    Charles McAnany

BPReveal 3.5.x
^^^^^^^^^^^^^^

Version 3.5.3, 2023-11-03
'''''''''''''''''''''''''

NEW FEATURES:
    * Added a bedUtils.py library with useful tools for manipulating bed files.

BUG FIXES:
    * Added fully qualified imports to several files, allowing you to import
      them from other directories.

CONTRIBUTORS:
    Charles McAnany

Version 3.5.2, 2023-10-26
'''''''''''''''''''''''''

NEW FEATURES:
    * Added a script to calculate the right counts loss weight given a model
      training history json.

BUG FIXES:
    * Fixed a typing bug in motifUtils that made motif scanning not work.

CONTRIBUTORS:
    Charles McAnany


Version 3.5.1, 2023-10-23
'''''''''''''''''''''''''

BUG FIXES:
    * Building conda environments is always haunted. Fixed problems with model
      training scripts not being able to find the cuda tools on Cerebro (even
      though they're found just fine on my local workstation!)

CONTRIBUTORS:
    Charles McAnany


Version 3.5.0, 2023-10-17
'''''''''''''''''''''''''

BREAKING CHANGES
    * This should not have any effects on typical uses, but BPReveal now uses
      Python 3.11.
    * Removed the compiled jaccard library, the install process now
      automatically builds it. You'll need to re-install BPReveal (or run make
      in the src directory).

NEW FEATURES:
    * Created a directory of helpful tools under src/tools. These are not part
      of BPReveal proper, but have some useful goodies for plotting and stuff.
      Pull requests welcome for new tools!
    * Added the ability to provide sequence fasta files to interpretFlat.py
      this required a total rewrite of the interpretation code to use
      streaming. interpretFlat now requires just a few gigs of memory. It also
      calculates profile and counts contribution simultaneously, leading to a
      60% speedup.
    * Created better conda integration. The BPReveal libraries should be on
      your python path when you open python, and they are in the bpreveal
      package. You can now `import bpreveal.utils` from any python interpreter.
      Also created a bin/ directory that has links to all of the BPReveal
      scripts. You should be able to just run `trainSoloModel config.json` once
      you've activated the conda environment.

ENHANCEMENTS:
    * Switched to storing importance scores as 16-bit floating point values and
      enabled hdf5 compression, leading to an 80% reduction in the size of
      contribution hdf5 files. Upgraded several components to effectively read
      and write in a compressed, block-oriented format.
    * Added type annotations to most of the library functions, allowing your
      editor to quickly check for mistakes in argument order and type.
    * Added type definitions to utils.py, so the code now (mostly) uses
      consistent definitions for variable types.
    * Updated the build scripts and added one for building a local copy of the
      BPReveal environment.

CONTRIBUTORS:
    Charles McAnany

BPReveal 3.4.x
^^^^^^^^^^^^^^

Version 3.4.0, 2023-10-06
'''''''''''''''''''''''''

NEW FEATURES:
    * CWM scanning is now implemented. This takes the output from modisco and
      uses contribution scores to look for motif instances across the genome.
      The documentation has been updated. Thanks to Melanie Weilert for an
      initial BPReveal-compatible implementation of CWM scanning.

CONTRIBUTORS:
    Melanie Weilert, Charles McAnany


BPReveal 3.3.x
^^^^^^^^^^^^^^

Version 3.3.2, 2023-09-19
'''''''''''''''''''''''''

BUG FIXES:
    * Updated the conda install script to be compatible with Tensorflow 2.12.
      The tensorflow-probability package that had been installed was too old,
      so I have changed to getting tensorflow and tensorflow-probability from
      conda. The build script also installs mamba, which seemed to work better
      for me.

CONTRIBUTORS:
    Charles McAnany


Version 3.3.1, 2023-08-30
'''''''''''''''''''''''''

ENHANCEMENTS:
    * Added a "heads" section to prepareBed.py json files. This lets you
      combined multiple bigwigs just as you do for the final model. The old
      "bigwigs" section is now deprecated, and will be removed in BPReveal 5.0.
      Previously, if you had a two-task head, prepareBed.py would reject any
      region where *either* of those tasks was outside of your counts limits.
      The new version adds the bigwigs in each head together before doing the
      counts culling. This is useful when one track has zero reads but the
      other still has data. Thanks to Melanie for suggesting this feature.
    * Finally ran through shap.py and fixed formatting.
    * Added two features to metrics.py. First, for regions that are empty,
      metrics.py now has a feature to simply ignore those regions rather than
      using them in counts correlations (they were always ignored in profile
      correlations). Second, added a feature to generate json output for ease
      of parsing.
    * Added three utility functions to gaOptimize.py for easily converting
      lists of corruptors to and from strings and numerical arrays. Thanks to
      Haining Jiang for suggesting these.

DEPRECATIONS:
    * The "bigwigs" section in prepareBed.py json files has been deprecated and
      will become an error in BPReveal 5.0.

CONTRIBUTORS:
    Melanie Weilert, Charles McAnany.


Version 3.3.0, 2023-06-23
'''''''''''''''''''''''''

NEW FEATURES:
    * Added a genetic algorithm module. See the demo pdfs for how to use them.
    * Added a batch-running tool to utils.py, this lets you run many sequences
      through your model without worrying about constructing batches
      efficiently.

ENHANCEMENTS:
    * Rewrote makePredictionsFasta to stream data in and out. It is now quite
      fast and uses very little memory.
    * Updated the OSKN demo python notebook to be compatible with version 3.

CONTRIBUTORS:
    Charles McAnany


BPReveal 3.2.x
^^^^^^^^^^^^^^

Version 3.2.0, 2023-05-17
'''''''''''''''''''''''''

NEW FEATURES:
    * Previously, if a solo model had a different input length than the
      residual model, you could not combine them. Melanie added logic so that
      if the solo model has a smaller input length (for example, because it has
      fewer layers), the sequence will automatically be cropped down to match
      it. In this way, you don't have to match solo and residual architectures
      any more.

ENHANCEMENTS:
    * Further re-formatting to comply with PEP8.

CONTRIBUTORS:
    Melanie Weilert (cropdown logic), Charles McAnany (code cleanup)

BPReveal 3.1.x
^^^^^^^^^^^^^^

Version 3.1.0, 2023-05-14
'''''''''''''''''''''''''

NEW FEATURES:
    * Added an automatic reverse complement strand selection feature. Instead
      of saying '"revcomp-task-order" : [1,0]', you can now say
      '"revcomp-task-order":"auto"' when you have one or two tasks in a head.

ENHANCEMENTS:
    * Code cleanup in general, such as removing unused imports and tidying up
      formatting.

BUG FIXES:
    * Fixed a missing import in prepareBed.py that broke the regex mode.

CONTRIBUTORS:
    Charles McAnany


BPReveal 3.0.x
^^^^^^^^^^^^^^

Version 3.0.1, 2023-04-26
'''''''''''''''''''''''''

ENHANCEMENTS:
    * Formatted the code throughout the repository to more closely comply with
      PEP8.

BUG FIXES:
    * Fixed a bug in argument order for deduplicating in prepareBed.py

CONTRIBUTORS:
    Charles McAnany

Version 3.0.0, 2023-03-10
'''''''''''''''''''''''''

BREAKING CHANGES:
    * There is a new "remove-overlaps" field that is mandatory in prepareBed.py
      json files. If set to true, then you can set how close two peaks must be
      before they are considered overlapping. (Thanks to Melanie Weilert for
      the implementation.)
    * On discussion with Melanie, it occurred that the cropdown feature of the
      transformation model is never appropriate. Therefore, this feature has
      been removed.  Instead, in a future version, there will be a feature to
      crop down the input sequence to the solo model during training the
      combined model. (Charles McAnany)
    * Since there is no cropping, it was silly to call the input-length
      "sequence-input-length" inside the transformation config json. It is now
      sensibly called "input-length".

ENHANCEMENTS:
    * The PISA code was totally rewritten; it now uses a streaming architecture
      so that loading the data, calculating shap scores, and saving data are
      done by different threads. This cuts way down on memory use, and makes it
      possible to run pisa over an entire genome. (generating 100 GiB per
      megabase or so.) (Charles McAnany)

BUG FIXES:
    * In the combined config, the documentation called a parameter
      "output-directory", but the code expected "output-prefix". The
      documentation has been corrected. (Charles McAnany)


CONTRIBUTORS:
    Melanie Weilert, Charles McAnany.

BPReveal 2.x
------------

BPReveal 2.0.x
^^^^^^^^^^^^^^

Version 2.0.2, 2023-02-17
'''''''''''''''''''''''''
ENHANCEMENTS:
    * interpretPisaBed.py will now include predictions and reference
      predictions in the output hdf5.

CONTRIBUTORS:
    Charles McAnany


Version 2.0.1, 2023-02-09
'''''''''''''''''''''''''
ENHANCEMENTS:
    * prepareBed.py will no longer replace the names in your bed files; the
      generated files will have the same names as the input beds. (Suggested by
      Melanie)

CONTRIBUTORS:
    Melanie Weilert, Charles McAnany


Version 2.0.0, 2023-02-07
'''''''''''''''''''''''''

BREAKING CHANGES:
    * Added a reverse-complement flag to prepareTrainingData.py. If this is set
      to true, then you must specify strand mappings to each of the heads in
      that file. If you want your code to behave like before, just set
      "reverse-complement" to false in the json file for
      prepareTrainingData.py.

ENHANCEMENTS:
    * Reverse complement support added, see overview.tex in the section on
      prepareTrainingData.py. (Charles McAnany)

CONTRIBUTORS:
    Charles McAnany.



PREVIOUS VERSIONS
-----------------

Versions of BPReveal before 2.0.0 are not recorded here, but the software
would not have been completed without help from Julia Zeitlinger, Anshul
Kundaje, and Melanie Weilert.


