
PISA
====

The traditional method of interpretation of BPNet models asks the question,
"How is it that this base here affects some readout over this window?",
where the readout is a single scalar value calculated over an entire prediction
window.
If our readout is counts, then we are calculating how much each input base
contributed to the total counts over our prediction window.
For profile, we calculate a weighted mean-normalized logit value (as was done in
the original BPNet paper).
In this way, each base gets a single scalar value for how it contributed to the
overall readout.

PISA asks a subtly different question:
"How is it that this base here affects the readout at that base over there?"
In this case, instead of looking at how much one base contributes to a global
readout, we're asking about its effect on a single one of its neighbors.
With PISA, each base gets a vector of its contributions to the readout at each
one of its neighbors.
This vector has shape ``(receptive-field,)``
Accordingly, if we perform PISA on a region of the genome, we would get a PISA
array of shape ``(region-length, receptive-field)``.

We use code derived from ``deepshap`` to perform this calculation.
With deepShap, we create an explainer that looks at one output from the model.
For the sake of simplicity, all of the PISA implementation looks at the leftmost
base in the output window, but this implementation detail has no effect on the
actual calculated values.
Once the explainer has been created, we provide it with genomic sequences to
explain, and it assigns, to each base in the input, its contribution to the
observed output.

The outputs of the model are logits, and the contribution scores have the same
units, so the explainer is effectively assigning a (base :math:`e`) fold-change value
to each input.

There are several properties of Shapley values that are important here.
In these formulae, :math:`\phi_i` is the contribution score of base :math:`i`, drawn from
the sequence :math:`S`.
The readout that we're measuring, the logit at the leftmost base, is :math:`v(S)`.
I'll also use :math:`K` to refer to a subset of the input sequence :math:`S`.
I'll use :math:`R` to refer to an ensemble of reference sequences, and the average
prediction from those reference sequences is :math:`\bar{v}(R)`

The first, and arguably most important, property of Shapley values is
*efficiency*:

.. math::

    \sum_{i \in S} \phi_i(v) = v(S) - \bar{v}(R)

This means that if we add up all the shap values that were assigned for a
particular logit, we recover the difference-from-reference of that logit.

One possible weakness of a method like this has to do with cooperation.
Suppose that some readout is observed if at least one of the bases in a region
is A.
If two bases are A, we observe that readout.
But how should we assign :math:`\phi` values to those two bases?
With shapley values, we are guaranteed that they will get the same score:

.. math::

    \Bigl( \forall (K \subseteq S \backslash \{i,j\})\;
    \bigl(v(K \cup \{i\}) = v(K \cup \{j\})\bigr) \Bigr)
    \\
    \implies \phi_i(v) = \phi_j(v)

The third property turns out to be very important in performing PISA on
bias-corrected models.
The combined model uses a simple sum to combine the logits from the solo model
and the residual model, so :math:`v_{combined}(S) = v_{solo}(S) + v_{residual}(S)`.
Shapley values preserve this linearity, ensuring that it's meaningful to look
at PISA plots of a residual model:

.. math::

    \phi_i(u + v) = \phi_i(u) + \phi_i(v)

Finally, an almost-obvious property: If a particular base has no effect on the
readout, its :math:`\phi` values should be zero. So it is:

.. math::

    \Bigl( \forall (K \subseteq S \backslash \{i\})
    \bigl( v(K \cup \{i\}) = v(K)\bigr)\Bigr)
    \\
    \implies \phi_i(v) = 0

..
    Copyright 2022-2025 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.
