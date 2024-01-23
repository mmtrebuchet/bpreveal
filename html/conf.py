# Configuration file for the Sphinx documentation builder.
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
release = '4.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
#              'sphinx_autodoc_typehints',
              'sphinx_rtd_theme',
              'sphinxarg.ext']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']





def setup(app):
    pass


autodoc_mock_imports = ["tensorflow", 'keras']
autodoc_typehints = 'both'
autodoc_type_aliases={
    "ANNOTATION_T": "ANNOTATION_T",
    "CorruptorLetter": "CorruptorLetter",
    "Corruptor": "Corruptor"
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
#html_theme = "alabaster"
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
