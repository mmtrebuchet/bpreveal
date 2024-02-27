
BNF Grammars
============

BPReveal specifies its inputs using a BNF grammar, since it's easier to read than
JSON schema and also quite precise.

A BNF grammar consists of names and definitions that define a language.

.. highlight:: none


Here is an example::

    <animal> ::=
        dog
      | cat

    <action> ::=
        pet
      | feed
      | play with

    <list-of-actions> ::=
        <action>
      | <action> and <list-of-actions>

    <sentence> ::=
        You should <list-of-actions> the <animal>.

This defines four terms: ``<animal>``, ``<action>``, ``<list-of-actions>``, and ``<sentence>``.
We can then use this to ask whether a string of letters is
a valid sentence. ``feed the dog`` would not be, but
``You should feed the cat.`` is, as is ``You should feed and pet and feed the dog``.

The pipe character, ``|`` indicates "or". So the term ``<animal>`` can be
either the string ``dog`` or the string ``cat``, but not both, and nothing else.

Anything that is not a pipe character or inside ``<`` angle brackets ``>`` is a literal.

Here are the base terms that are used throughout the documentation:

.. include:: bnf/base.rst

