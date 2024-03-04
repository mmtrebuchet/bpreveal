Release Checklist
=================

    1. Update version in __init__.py

    2. pylint, pydocstyle, flake8

    3. Changelog, breaking changes.

    4. Make in src/

    5. Make in doc/

    6. Run OSKN model up through PISA. Render notebook.

    7. git add, git commit

    8. git checkout master, merge.

    9. git tag

    10. git push

    11. git push --tags

    12. git branch

    13. git clone in public-bpreveal

    14. Build conda environment on Cerebro.

    15. Symlinks in public-bpreveal, including documentation directory.

    16. Copy documentation to mmtrebuchet.github.io, edit both index.html files.

    17. git add, git commit, git push in mmtrebuchet.github.io.

    18. Issue release on github with pdf of documentation.

    19. (major and minor releases only) Announce on Teams.

..
    Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa  # pylint: disable=line-too-long
