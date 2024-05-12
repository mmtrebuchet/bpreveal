Release Checklist
=================

    1. Update version in __init__.py

    2. pylint, pydocstyle, flake8

    3. Changelog, breaking changes. Set date in changelog.

    4. git add, git commit

    5. git clone in public-bpreveal

    6. Build conda environment on Cerebro.

    7. Make in doc/

    8. Run OSKN model up through PISA. Render notebook. Commit the final notebook.

    9. Symlinks in public-bpreveal, including documentation directory.

    10. git checkout master, merge.

    11. git tag

    12. git push

    13. git push --tags

    14. git branch

    15. Issue release on github with pdf of documentation.

    16. (major and minor releases only) Announce on Teams.

..
    Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.
