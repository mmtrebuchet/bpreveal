# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
export COLUMNS = 60
export LATEXMKOPTS="-interaction=batchmode -halt-on-error "
# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# We're about to include a header file that gets made by
# build.py. Create a rule for what to do if the header is missing...
_generated/makeHeader: build.py
	./build.py

#And then include it.
include _generated/makeHeader

gen: build.py ${allGenerated}

clean:
	if [ -d "_generated" ]; then rm -r _generated; fi
	if [ -d "_build" ]; then rm -r _build; fi

html: Makefile ${allGenerated}
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
latexpdf: Makefile ${allGenerated}
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

