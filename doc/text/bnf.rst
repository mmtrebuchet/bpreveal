
BNF Grammars
============

BPReveal specifies its inputs using a BNF grammar, since it's easier to read than
JSON schema and also quite precise.

BPReveal uses a custom interpreter to read in the configuration files, this interpreter
implements a superset of JSON. Here are the added features:

* List comprehensions, using normal Python syntax.
* Dictionary comprehensions, using normal Python syntax.
* Comments, which start with ``#`` and go to the end of the line.
* Python arithmetic, boolean, and comparison operators.
* Function calls to Python essentials like range and len
* Lambda support, with the ability to use default arguments to define recursive
  procedures à la ``letrec``
* Ternary if expressions.

See :py:mod:`interpreter<bpreveal.internal.interpreter>` for a full
specification. The extension is Turing complete, but all of the functions are
pure and IO is not possible. Therefore you can run it on untrusted code (with
the attendant risk of stack overflows and DoS attacks from runaway processes).


A BNF grammar consists of names and definitions that define a language.

.. highlight:: none


Here is an example::

    <animal> ::=
        dog
      | cat

    <possibly-plural-animal> ::=
        <animal>«s»

    <action> ::=
        pet
      | feed
      | play with

    <list-of-actions> ::=
        <action>« and <list-of-actions>»

    <sentence> ::=
        You should <list-of-actions> the <possibly-plural-animal>.

This defines five terms: ``<animal>``, ``<possibly-plural-animal>``,
``<action>``, ``<list-of-actions>``, and ``<sentence>``.
We can then use this to ask whether a string of letters is
a valid sentence. ``feed the dog`` would not be, but
``You should feed the cat.`` is, as is ``You should feed and pet and feed the dogs``.

The pipe character, ``|`` indicates "or". So the term ``<animal>`` can be
either the string ``dog`` or the string ``cat``, but not both, and nothing else.

Double brackets (``«»``) indicate that whatever is inside is optional.
So a ``<possibly-plural-animal>`` is defined to be an ``<animal>`` followed
by an optional letter ``s``.

Anything that is not a pipe character, a double bracket, or
inside ``<`` angle brackets ``>`` is a literal.

Here are the base terms that are used throughout the documentation:

.. include:: bnf/base.rst

..
    Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.
