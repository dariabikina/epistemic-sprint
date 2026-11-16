# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Ground-truth worlds: small transformers trained on sequences whose Bayes-optimal prediction is known exactly.

In a real language model we never know what the model *should* believe. Here we do: every sequence is drawn
from a stated prior (a "world"), and for every prefix the posterior predictive (the best any predictor could
do) and its split into irreducible noise (aleatoric) and resolvable ignorance (epistemic) can be computed.
"Knowing what it knows" becomes measurable against truth.

    priors.py   the worlds: samplers and exact Bayes predictives, from a constant level up to Zipfian symbols
    data.py     frozen pools (train, validation with Bayes answers), sha256 manifest
    train.py    a small Llama trained from scratch on one world; validation against the Bayes floor
    cli.py      `gt data | train | ladder | show | plot | smoke`
"""
