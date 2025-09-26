Release Checklist
=================

    1. Update version in __init__.py, etc/buildPip.zsh, pyproject.toml

    2. pylint, pydocstyle, flake8

    3. Changelog, breaking changes. Set date in changelog.

    4. make schemas

    5. git add, git commit

    6. git clone in public-bpreveal as a release candidate.

    7. Build conda environment on Cerebro.

    8. Run OSKN acceptance test.

    9. git checkout master, merge.

    10. git tag

    11. git push

    12. git push --tags

    13. Build release environment on Cerebro.

    14. Make in doc/ on Cerebro.

    15. Symlinks in public-bpreveal, including documentation directory.

    16. etc/buildPip.zsh

    17. twine upload to testpypi.

    18. pip install from test repo

    19. twine upload to pypi

    20. Issue release on github with pdf of documentation and wheel.

    21. (major and minor releases only) Announce on Teams.

    22. git branch

..
    Copyright 2022-2025 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.
