#!/usr/bin/env python3
"""Make a PISA plot or graph (depending on the input json).

BNF
---

.. highlight:: none

.. literalinclude:: ../../doc/bnf/makePisaFigure.bnf

Parameter notes
---------------

``graph-configs``, ``plot-configs``
    A list of configurations appropriate for the functions in
    :py:mod:`plotting<bpreveal.plotting>`. At least one of these
    must be present, and you can have both.

``output-png``, ``output-pdf``
    The name you want your figure saved as. You can specify both in order
    to have both a pdf and png saved of the same figure.
    At least one must be specified, and you can have both.

``width``, ``height``
    (optional) The width and height of the generated figure, in inches.
    Default: width 7, height 5.

``dpi``
    (optional) Only relevant when you render to a png, this determines the resolution.
    Default: 300

``transparent``
    (optional) Should the figure have a transparent background?
    Default: False

"""

import sys
import time
import json
import matplotlib.pyplot as plt
from bpreveal import plotting
from bpreveal import logUtils


def main(cfg):
    """Actually make the plot(s)."""
    startTime = time.perf_counter()
    logUtils.setVerbosity(cfg.get("verbosity", "WARNING"))
    dpi = cfg.get("dpi", 300)
    width = cfg.get("width", 7)
    height = cfg.get("height", 5)
    transparent = cfg.get("transparent", False)
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    logUtils.info("Starting to draw graphs.")
    for c in cfg.get("graph-configs", []):
        # We want a pisa graph.
        plotting.plotPisaGraph(c, fig)
    logUtils.info("Starting to draw plots.")
    for c in cfg.get("plot-configs", []):
        # We want a pisa graph.
        plotting.plotPisa(c, fig)
    logUtils.info("Saving.")
    if "output-png" in cfg:
        fig.savefig(cfg["output-png"], dpi=dpi, transparent=transparent)
    if "output-png" in cfg:
        fig.savefig(cfg["output-png"], dpi=dpi, transparent=transparent)
    Δt = time.perf_counter() - startTime
    logUtils.info(f"Figure generation complete in {Δt}.")


if __name__ == "__main__":
    with open(sys.argv[1], "r") as cfgFp:
        config = json.load(cfgFp)
    main(config)
