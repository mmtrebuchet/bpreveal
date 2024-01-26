"""Configuration file for the Sphinx documentation builder."""
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

project = 'BPReveal'
copyright = '2024, Charles McAnany, Melanie Weilert, Haining Jiang'
author = 'Charles McAnany, Melanie Weilert, Haining Jiang'
import bpreveal
release = bpreveal.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
#              'sphinx_autodoc_typehints',
              'sphinx_rtd_theme',
              'sphinxarg.ext']
#              'sphinxcontrib.restbuilder']

templates_path = []
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "text", "bnf",
                    "demos", "presentations", "scripts",
                    "makeHeader"]
import re
reSubs = [
    ["λ", r":math:`{\\lambda}`"],
    ["½", r":math:`{1/2}`"]]

def fixLambda(app, what, name, obj, options, lines):

    for i, line in enumerate(lines):
        for rsub, rrepl in reSubs:
            line = re.sub(rsub, rrepl, line)
        lines[i] = line


def setup(app):
    app.connect('autodoc-process-docstring', fixLambda)
    pass


autodoc_mock_imports = ["tensorflow", 'keras']
autodoc_typehints = 'both'
autodoc_member_order = 'bysource'
autodoc_type_aliases={
    "ANNOTATION_T": "ANNOTATION_T",
    "CorruptorLetter": "CorruptorLetter",
    "Corruptor": "Corruptor",
    "CandidateCorruptor" : "CandidateCorruptor",
    "Profile": "Profile"
}
autodoc_unqualified_typehints = False
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
#html_theme = "alabaster"
html_theme = "sphinx_rtd_theme"
html_static_path = []
html_css_files = []


