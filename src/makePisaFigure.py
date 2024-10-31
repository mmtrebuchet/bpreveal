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
    The name you want your figure saved as. You can specify both in order to
    have both a pdf and png saved of the same figure.
    If you specify both ``output-gui`` and also save the figure with
    ``output-png`` or ``output-pdf``, then the figure will be saved
    *before* any manipulation in the GUI.

``width``, ``height``
    (optional) The width and height of the generated figure, in inches.
    Default: width 7, height 5.

``dpi``
    (optional) Only relevant when you render to a png, this determines the resolution.
    Default: 300

``transparent``
    (optional) Should the figure have a transparent background?
    Default: False

``output-gui``
    (optional) Should the figure be displayed with plt.show()?
    Default: False.

The configuration dictionaries for the individual plots and graphs are detailed at
:py:mod:`plotting<bpreveal.plotting>`.
"""

import sys
import datetime
import time
import matplotlib.pyplot as plt
from bpreveal import plotting
from bpreveal import logUtils
import bpreveal
from bpreveal.internal import interpreter
from bpreveal import colors


def main(cfg: dict) -> None:
    """Actually make the plot(s)."""
    startTime = time.perf_counter()
    colors.loadFonts()
    logUtils.setVerbosity(cfg.get("verbosity", "WARNING"))
    dpi = cfg.get("dpi", 300)
    width = cfg.get("width", 7)
    height = cfg.get("height", 5)
    transparent = cfg.get("transparent", False)
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    logUtils.info("Starting to draw graphs.")
    metadata = {"bpreveal_version": bpreveal.__version__,
                "config": str(cfg),
                "created_date": str(datetime.datetime.today())
                }
    for c in cfg.get("graph-configs", []):
        # We want a pisa graph.
        plotting.plotPisaGraph(c, fig)
    logUtils.info("Starting to draw plots.")
    for c in cfg.get("plot-configs", []):
        # We want a pisa graph.
        plotting.plotPisa(c, fig)
    logUtils.info("Saving.")
    if "output-png" in cfg:
        fig.savefig(cfg["output-png"], dpi=dpi, transparent=transparent,
                    metadata=metadata)
    if "output-png" in cfg:
        fig.savefig(cfg["output-png"], dpi=dpi, transparent=transparent,
                    metadata=metadata)
    Δt = time.perf_counter() - startTime
    logUtils.info(f"Figure generation complete in {Δt}.")
    if cfg.get("output-gui", False):
        logUtils.info("Starting GUI.")
        plt.show()


if __name__ == "__main__":
    config = interpreter.evalFile(sys.argv[1])
    assert isinstance(config, dict)
    main(config)
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
