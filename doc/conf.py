"""Configuration file for the Sphinx documentation builder."""
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import re
import bpreveal
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

project = 'BPReveal'
copyright = '2024, Charles McAnany, Melanie Weilert, Haining Jiang'
author = 'Charles McAnany, Melanie Weilert, Haining Jiang'
release = bpreveal.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              #              'sphinx_autodoc_typehints',
              'sphinx_rtd_theme',
              'sphinxarg.ext']
#              'sphinxcontrib.restbuilder']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "text", "bnf",
                    "demos", "presentations", "scripts",
                    "makeHeader"]
reSubs = [
    ["λ", r":math:`{\\lambda}`"],
    ["½", r":math:`{1/2}`"]]


def fixLambda(app, what, name, obj, options, lines):
    """Replace greek and other unicode letters in docstrings with something that TeX can handle."""
    for i, line in enumerate(lines):
        for rsub, rrepl in reSubs:
            line = re.sub(rsub, rrepl, line)
        lines[i] = line


def setup(app):
    """Connect the lambda corrector."""
    app.connect('autodoc-process-docstring', fixLambda)
    pass


autodoc_mock_imports = ["tensorflow", 'keras']
autodoc_typehints = 'both'
autodoc_member_order = 'bysource'
autodoc_type_aliases = {
    "ANNOTATION_T": "ANNOTATION_T",
    "CorruptorLetter": "CorruptorLetter",
    "Corruptor": "Corruptor",
    "CandidateCorruptor": "CandidateCorruptor",
    "Profile": "Profile"
}
autodoc_unqualified_typehints = False
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
#html_theme = "alabaster"
templates_path = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_generated/static"]
html_css_files = ["custom-styles.css", "libertinus.css"]
