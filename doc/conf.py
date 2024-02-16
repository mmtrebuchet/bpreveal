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
# pylint: disable=invalid-name
project = 'BPReveal'
copyright = '2024, Charles McAnany, Melanie Weilert, Haining Jiang'  # pylint: disable=redefined-builtin # noqa
author = 'Charles McAnany, Melanie Weilert, Haining Jiang'
release = bpreveal.__version__
# pylint: enable=invalid-name

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx_rtd_theme',
              'sphinxarg.ext']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "text", "bnf",
                    "demos", "presentations", "scripts", "_generated/bnf/base.rst",
                    "makeHeader"]
reSubs = [
    ["λ", r":math:`{\\lambda}`"],
    ["½", r":math:`{1/2}`"],
    ["∬", r":math:`{\\iint}`"],
    [r".. literalinclude:: ../../doc/bnf/(.*).bnf",
     r".. include:: ../../doc/_generated/bnf/\1.rst"]
]


def fixLambda(app, what, name, obj, options, lines):  # pylint: disable=unused-argument
    """Replace Greek and other unicode letters in docstrings with something that TeX can handle."""
    for i, line in enumerate(lines):
        for rsub, rrepl in reSubs:
            line = re.sub(rsub, rrepl, line)
        lines[i] = line


def setup(app):
    """Connect the lambda corrector."""
    app.connect('autodoc-process-docstring', fixLambda)


autodoc_mock_imports = ["tensorflow", 'keras']
autodoc_typehints = 'both'  # pylint: disable=invalid-name
autodoc_member_order = 'bysource'  # pylint: disable=invalid-name
autodoc_type_aliases = {
    "ANNOTATION_T": "ANNOTATION_T",
    "CorruptorLetter": "CorruptorLetter",
    "Corruptor": "Corruptor",
    "CandidateCorruptor": "CandidateCorruptor",
    "Profile": "Profile"
}
# pylint: disable=invalid-name
autodoc_unqualified_typehints = False
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# html_theme = "alabaster"
templates_path = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_generated/static"]
html_css_files = ["custom-styles.css", "libertinus.css"]
default_dark_mode = False
# pylint: enable=invalid-name
