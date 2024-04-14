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
copyright = '2024, Charles McAnany'  # pylint: disable=redefined-builtin # noqa
authorList = bpreveal.__author__.split(';')
authorList = [x.strip() for x in authorList]
author = ", ".join(authorList)
show_authors = True
release = bpreveal.__version__
version = release
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
    ["Δ", r":math:`{\\Delta}`"],
    ["½", r":math:`{1/2}`"],
    ["∬", r":math:`{\\iint}`"],
    [r".. literalinclude:: ../../doc/bnf/(.*).bnf",
     r".. include:: ../../doc/_generated/bnf/\1.rst"]
]

for definedType in ["ONEHOT_T", "ONEHOT_AR_T", "PRED_T", "PRED_AR_T",
                    "LOGIT_T", "LOGIT_AR_T", "LOGCOUNT_T",
                    "IMPORTANCE_T", "IMPORTANCE_AR_T", "MODEL_ONEHOT_T",
                    "MOTIF_FLOAT_T"]:
    reSubs.append(
        [definedType, f" :py:data:`{definedType}<bpreveal.internal.constants.{definedType}>` "])


def fixLambda(app, what, name, obj, options, lines):  # pylint: disable=unused-argument
    """Replace Greek and other unicode letters in docstrings with something that TeX can handle."""
    for i, line in enumerate(lines):
        for rsub, rrepl in reSubs:
            line = re.sub(rsub, rrepl, line)
        lines[i] = line


def setup(app):
    """Connect the lambda corrector."""
    app.connect('autodoc-process-docstring', fixLambda)


autodoc_mock_imports = ["tensorflow", 'keras',
                        "tensorflow_probability"]
autodoc_typehints = 'description'  # pylint: disable=invalid-name
autodoc_member_order = 'bysource'  # pylint: disable=invalid-name
autodoc_type_aliases = {
    "ANNOTATION_T": "ANNOTATION_T",
    "ONEHOT_T": "ONEHOT_T",
    "CorruptorLetter": "CorruptorLetter",
    "Corruptor": "Corruptor",
    "CandidateCorruptor": "CandidateCorruptor",
    "Profile": "Profile"
}
texAuthors = r" \and ".join(authorList)
latex_documents = [
    ("index",
    "bpreveal.tex",
    "BPReveal",
    texAuthors,
    "manual",
    False)
]
latex_elements = {
    "preamble": r"""\DeclareRobustCommand{\and}{%
\end{tabular}\kern-\tabcolsep\\\begin{tabular}[t]{c}%
}%
""",
}
# pylint: disable=invalid-name
autodoc_unqualified_typehints = True
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# html_theme = "alabaster"
templates_path = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_generated/static"]
html_css_files = ["custom-styles.css", "libertinus.css"]

# pylint: enable=invalid-name
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa  # pylint: disable=line-too-long
