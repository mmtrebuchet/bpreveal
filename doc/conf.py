"""Configuration file for the Sphinx documentation builder."""
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import re
import os
import bpreveal
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# pylint: disable=invalid-name, redefined-builtin
project = "BPReveal"
copyright = "2024, Charles McAnany. " \
    "License: GPLv2+: GNU GPL version 2 or later " \
    "<https://gnu.org/licenses/gpl.html>. " \
    "This is free software: you are free to change and redistribute it. " \
    "There is NO WARRANTY, to the extent permitted by law."
authorList = bpreveal.__author__.split(";")
authorList = [x.strip() for x in authorList]
author = ", ".join(authorList)
show_authors = True
release = bpreveal.__version__
version = release
# pylint: enable=invalid-name, redefined-builtin

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc",
              "sphinx_rtd_theme",
              "sphinxarg.ext"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "text", "bnf",
                    "demos", "presentations", "scripts", "_generated/bnf/base.rst",
                    "makeHeader"]

man1 = ["addNoise", "bestMotifsOnly", "checkJson", "interpretFlat",
        "interpretPisa", "lengthCalc", "lossWeights", "makeLossPlots",
        "makePisaFigure", "makePredictions", "metrics", "motifAddQuantiles",
        "motifScan", "motifSeqletCutoffs", "predictToBigwig", "prepareBed",
        "prepareTrainingData", "revcompTools", "shapToBigwig", "shapToNumpy",
        "shiftBigwigs", "shiftPisa", "showModel", "showTrainingProgress",
        "tileGenome", "trainSoloModel", "trainTransformationModel",
        "trainCombinedModel", "filterProc"]
man3 = ["addNoiseUtils", "bedUtils", "callbacks", "colors", "constants",
        "disableTensorflowLogging", "files", "gaOptimize", "generators",
        "interpretUtils", "jaccard", "layers", "logUtils", "losses", "models",
        "motifUtils", "plotting", "plotUtils", "predictUtils", "schema",
        "slurm", "training", "ushuffle", "utils"]
man7 = ["bnf", "breakingChanges", "changelog", "countsLossReweighting",
        "license", "modelArchitectures", "philosophy", "pisa", "programs",
        "releaseChecklist", "setup", "workflow"]
reSubs = [
    ["λ", r":math:`{\\lambda}`", ["MAN_PAGE"]],
    ["Δ", r":math:`{\\Delta}`", ["MAN_PAGE"]],
    ["½", r":math:`{1/2}`", ["MAN_PAGE"]],
    ["∬", r":math:`{\\iint}`", ["MAN_PAGE"]],
    [r".. literalinclude:: ../../doc/bnf/(.*).bnf",
     r".. include:: ../../doc/_generated/bnf/\1.rst", []]
]

for definedType in ("ONEHOT_T", "ONEHOT_AR_T", "PRED_T", "PRED_AR_T",
                    "LOGIT_T", "LOGIT_AR_T", "LOGCOUNT_T",
                    "IMPORTANCE_T", "IMPORTANCE_AR_T", "MODEL_ONEHOT_T",
                    "MOTIF_FLOAT_T"):
    reSubs.append(
        [definedType, f" :py:data:`{definedType}<bpreveal.internal.constants.{definedType}>` ", []])

for pages, section in ((man1, 1), (man3, 3), (man7, 7)):
    curStr = ""  # pylint: disable=invalid-name
    curStr += ", ".join(pages)
    reSubs.append([f"MAN_PAGES_SECTION_{section}", curStr, []])


def preprocessLines(lines: list[str], defs: list[str]) -> None:
    """Run the macro preprocessor over the lines of text in lines."""
    printing = [True]
    for i, line in enumerate(lines):
        match line.split():
            case ("#define", symbol):
                defs.append(symbol)
            case ("#undef", symbol):
                defs = [x for x in defs if x != symbol]
            case ("#ifdef", symbol):
                printing.append(symbol in defs)
                lines[i] = ""
            case ("#else",):
                printing[-1] = not printing[-1]
                lines[i] = ""
            case ("#endif",):
                printing = printing[:-1]
                lines[i] = ""
            case ("#ifndef", symbol):
                printing.append(symbol not in defs)
                lines[i] = ""
            case _:
                if printing[-1]:
                    for rsub, rrepl, skipEnvs in reSubs:
                        skip = False
                        for se in skipEnvs:
                            if se in defs:
                                skip = True
                        if not skip:
                            line = re.sub(rsub, rrepl, line)  # pylint: disable=redefined-loop-name
                    lines[i] = line
                else:
                    lines[i] = ""

# pylint: disable=unused-argument
def fixLambda(app, what, name, obj, options,  # noqa
              lines: list[str]) -> None:  # noqa
    """Replace Greek and other unicode letters in docstrings with something that TeX can handle."""
    defs = []
    if app.outdir.parts[-1] == "man":
        defs.append("MAN_PAGE")
    preprocessLines(lines, defs)


def sourceRead(app, docname: str, source: list[str]) -> None:  # noqa
    """Read in a .rst file and run the preprocessor."""
    lines = source[0].split("\n")
    defs = []
    if app.outdir.parts[-1] == "man":
        defs.append("MAN_PAGE")
    preprocessLines(lines, defs)
    source[0] = "\n".join(lines)

# pylint: enable=unused-argument


def setup(app) -> None:  # noqa: ANN001
    """Connect the lambda corrector."""
    app.connect("autodoc-process-docstring", fixLambda)
    app.connect("source-read", sourceRead)


autodoc_mock_imports = ["tensorflow", "keras", "tf_keras",
                        "tensorflow_probability"]
autodoc_typehints = "description"  # pylint: disable=invalid-name
autodoc_member_order = "bysource"  # pylint: disable=invalid-name
autodoc_type_aliases = {
    "ANNOTATION_T": "ANNOTATION_T",
    "ONEHOT_T": "ONEHOT_T",
    "CorruptorLetter": "CorruptorLetter",
    "Corruptor": "Corruptor",
    "CandidateCorruptor": "CandidateCorruptor",
    "Profile": "Profile"
}
TEX_AUTHORS = r" \and ".join(authorList)
latex_documents = [
    ("index",
    "bpreveal.tex",
    "BPReveal",
    TEX_AUTHORS,
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
templates_path = []
html_theme = "sphinx_rtd_theme"  # Was "alabaster" originally.
html_static_path = ["_generated/static"]
html_css_files = ["custom-styles.css", "libertinus.css"]

man_make_section_directory = True
man_pages = []
for mp in man1 + man3 + man7:
    idx = None
    if mp in man1:
        idx = 1
    elif mp in man3:
        idx = 3
    elif mp in man7:
        idx = 7
    man_pages.append((
        "_generated/" + mp,
        mp,
        "",
        author,
        idx))
man_pages.append(("index", "bpreveal", "", author, 7))
# pylint: enable=invalid-name
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa  # pylint: disable=line-too-long
