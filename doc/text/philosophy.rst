Developer's Manifesto
=====================

Coding philosophy
-----------------

*   Make each program do one thing well. To do a new job, build afresh rather
    than complicate old programs by adding new features.
*   Trust the user. Don't add validity checks except for instances where you
    are certain that the user is making a mistake that will pass undetected.
*   Expect the output of every program to become the input to another,
    as yet unknown, program.
*   Don't insist on interactive input.
*   Don't clutter the output with extraneous information.
*   But do include logging messages that can be enabled for debugging.
*   Explicit is better than implicit.
    Wherever possible, do not allow for defaults when a value is not
    specified. (Looking at you, MoDISco!)
*   For substantial programs, prefer configuration files over a sea of
    command-line arguments.
    Use JSON for all data that aren't (1.) bed files, (2.) bigwig files,
    (3.) fasta files, or (4.) potentially enormous
    (for large datasets, prefer hdf5.).
*   Don't be too clever.
    The code should use the standard idioms of the language, even if an
    operation could be completed in fewer characters or slightly more
    efficiently some other way.
*   But performance matters.
    Write code that works fast enough that people can use it to ask new
    questions.
*   Use parallelism with wild abandon.
*   Errors should never pass silently.
*   One line of code is ten lines of documentation.
    The documentation consists of in-code comments, clear specifications
    (this documentation), tutorials, and the reference publication.
*   If the implementation is hard to explain, it's a bad idea.
*   Use only well-established and stable dependencies.
    Don't require specific versions of libraries, and only require packages
    that are truly essential.
*   WE DO NOT BREAK USERSPACE!
    Do not introduce backwards-incompatible changes unless they are necessary.
    If a configuration format changes, allow both old-style and new-style
    configurations so that existing code continues to work correctly.


Coding Standards
----------------

As of 4.1.0, BPReveal gets a perfect score from pylint, flake8, and pydocstyle.

The integration test includes coverage testing, and new features should always
be added to the integration test suite.

BPReveal uses normal semantic versioning. Major versions may introduce
backwards-incompatible behavior to the CLI and any part of the API. Only
major versions can alter the structure of files on disk.
Minor versions may introduce breaking changes only to parts of the API
that are not intended for external use.
Patches are only for bug fixes.



..
    Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.
