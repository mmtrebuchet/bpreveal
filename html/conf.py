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
              #'sphinx.ext.autosummary',
              #'sphinx_automodapi.automodapi',
              'sphinx_autodoc_typehints',
              'sphinx_rtd_theme',
              'sphinxarg.ext']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
from sphinx_automodapi import automodsumm
from sphinx_automodapi.utils import find_mod_objs

#def find_mod_objs_patched(*args, **kwargs):
#    return find_mod_objs(args[0], onlylocals=True)

#def patch_automodapi(app):
#    """Monkey-patch the automodapi extension to exclude imported members"""
#    automodsumm.find_mod_objs = find_mod_objs_patched



def showSkips(app, what, name, obj, skip, options):
    if name == "__init__":
        return False
    return skip

def setup(app):
 #   app.connect("builder-inited", patch_automodapi)
    app.connect("autodoc-skip-member", showSkips)


automodapi_inheritance_diagram = False
automodapi_writepreprocessed = True
autodoc_mock_imports = ["tensorflow", 'keras']
autodoc_typehints = 'both'
autodoc_type_aliases={"ANNOTATION_T": "bpreveal.gaOptimize.ANNOTATION_T"}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
