# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -E
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     ?= .
BUILDDIR      ?= _build
export COLUMNS = 60
export LATEXMKOPTS="-interaction=batchmode -halt-on-error "
CSS_NAMES = custom-styles libertinus
FONT_NAMES = Keyboard-Regular Math-Regular Mono-Regular \
			 Sans-Bold Sans-Italic Sans-Regular \
			 Serif-BoldItalic Serif-Bold Serif-Italic Serif-Regular \
			 SerifDisplay-Regular \
			 SerifInitials-Regular \
			 Serif-SemiboldItalic Serif-Semibold

fonts = $(patsubst %,$(BUILDDIR)/html/_static/Libertinus%.woff2, ${FONT_NAMES})
css_files = $(patsubst %,_generated/static/%.css,${CSS_NAMES})

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	if [ -d "_generated" ]; then rm -r _generated; fi
	if [ -d "$(BUILDDIR)" ]; then rm -r $(BUILDDIR); fi
	if [ -f index.rst ]; then rm index.rst; fi

.PHONY: help Makefile

# We're about to include a header file that gets made by
# build.py. Create a rule for what to do if the header is missing...
_generated/makeHeader: build.py
	./build.py make

#And then include it.
include _generated/makeHeader

gen: build.py ${allGenerated} conf.py


noregen: Makefile conf.py
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

$(css_files): _generated/static/%.css: text/%.css
	cp $< $@

$(fonts): $(BUILDDIR)/html/_static/%.woff2: fonts/Libertinus-7.040/static/%.woff2
	cp $< $@

nofonthtml: Makefile ${allGenerated} conf.py ${css_files}
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

#	cp -R fonts/Libertinus-7.040/static/* _build/html/_static

html: nofonthtml ${fonts}

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
latexpdf: Makefile ${allGenerated} conf.py
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


man: Makefile ${allGenerated} conf.py
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
