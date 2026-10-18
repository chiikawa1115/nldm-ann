"""PLAN v4 W2: inference-available encoding of the timing arc (when, related pin, sense).

ASAP7 `function` and `when` strings are sum-of-products over input pins, e.g.
`(A1 * A2) + (B)` and `(A2 * !B)`. Pins are put in a canonical order (letter
group, then index: A1 < A2 < B < C1), which follows the Boolean grouping of the
cell, and every arc is described relative to that order:

- when slots: for each canonical pin, the mean value the `when` condition
  requires over its product terms (1, 0, or 0.5 for don't-care); the related
  pin itself is -1 and unused slots are -2. Together with the number of `when`
  terms this removes every feature conflict in the mainline (W2 check).
- pin structure: type-independent descriptors of the related pin (Boolean
  influence, number/length of product terms containing it, its group size), so
  a family-level model (L2) sees comparable meaning across types.

No pin name string, cell ID or type ID is used as a feature.
"""

from __future__ import annotations

import itertools
import re
from functools import lru_cache

import numpy as np
import pandas as pd

MAX_PINS = 9  # AO333 / OA333 have nine inputs; asserted when encoding
TIMING_SENSES = ("positive_unate", "negative_unate", "non_unate")
WHEN_SLOT_DONT_CARE = 0.5
WHEN_SLOT_SELF = -1.0
WHEN_SLOT_UNUSED = -2.0
_PIN_RE = re.compile(r"^([A-Z]+?)(\d*)$")


def parse_sop(expression: str) -> tuple[dict[str, int], ...]:
    """'(A * !B) + (C)' -> ({'A': 1, 'B': 0}, {'C': 1}); empty string -> ()."""
    text = str(expression or "").strip()
    if not text:
        return ()
    terms = []
    for term in text.split("+"):
        literals = [lit.strip() for lit in term.strip().strip("()").split("*") if lit.strip()]
        if not literals:
            raise ValueError(f"empty product term in {expression!r}")
        parsed = {}
        for lit in literals:
            name = lit.lstrip("!").strip()
            if not _PIN_RE.match(name):
                raise ValueError(f"unsupported literal {lit!r} in {expression!r}")
            parsed[name] = 0 if lit.startswith("!") else 1
        terms.append(parsed)
    return tuple(terms)


def pin_sort_key(pin: str) -> tuple[str, int]:
    match = _PIN_RE.match(pin)
    if not match:
        raise ValueError(f"unsupported pin name {pin!r}")
    return match[1], int(match[2] or 0)


def canonical_pins(function: str) -> tuple[str, ...]:
    pins = sorted({v for term in parse_sop(function) for v in term}, key=pin_sort_key)
    if len(pins) > MAX_PINS:
        raise ValueError(f"{len(pins)} inputs exceed MAX_PINS={MAX_PINS}: {function!r}")
    return tuple(pins)


def _evaluate(terms, assignment) -> bool:
    return any(all(assignment[v] == b for v, b in term.items()) for term in terms)


@lru_cache(maxsize=None)
def pin_structure(function: str, pin: str) -> tuple[float, ...]:
    """(influence, n_terms_with_pin, mean_term_len, group_size, n_terms, n_inputs)."""
    terms = parse_sop(function)
    pins = canonical_pins(function)
    if pin not in pins:
        raise ValueError(f"related pin {pin!r} not in function {function!r}")
    others = [p for p in pins if p != pin]
    flips = 0
    for bits in itertools.product((0, 1), repeat=len(others)):
        assignment = dict(zip(others, bits))
        assignment[pin] = 0
        low = _evaluate(terms, assignment)
        assignment[pin] = 1
        flips += _evaluate(terms, assignment) != low
    own = [t for t in terms if pin in t]
    group = pin_sort_key(pin)[0]
    return (
        flips / 2 ** len(others),
        float(len(own)),
        float(np.mean([len(t) for t in own])),
        float(sum(pin_sort_key(p)[0] == group for p in pins)),
        float(len(terms)),
        float(len(pins)),
    )


@lru_cache(maxsize=None)
def when_slots(function: str, pin: str, when: str) -> tuple[float, ...]:
    """MAX_PINS slot values followed by the number of `when` product terms."""
    pins = canonical_pins(function)
    terms = parse_sop(when)
    unknown = {v for t in terms for v in t} - set(pins)
    if unknown:
        raise ValueError(f"when {when!r} uses pins outside function {function!r}: {unknown}")
    if any(pin in t for t in terms):
        raise ValueError(f"when {when!r} constrains the related pin {pin!r}")
    slots = []
    for p in pins:
        if p == pin:
            slots.append(WHEN_SLOT_SELF)
        elif terms:
            slots.append(float(np.mean([t.get(p, WHEN_SLOT_DONT_CARE) for t in terms])))
        else:
            slots.append(WHEN_SLOT_DONT_CARE)
    slots += [WHEN_SLOT_UNUSED] * (MAX_PINS - len(pins))
    return (*slots, float(len(terms)))


PIN_COLUMNS = ["pin_influence", "pin_n_terms", "pin_term_len", "pin_group_size",
               "func_n_terms", "func_n_inputs"]
WHEN_COLUMNS = [f"when_slot{i}" for i in range(MAX_PINS)] + ["when_n_terms"]
SENSE_COLUMNS = [f"sense_{s}" for s in TIMING_SENSES]
ARC_FEATURE_COLUMNS = PIN_COLUMNS + WHEN_COLUMNS + SENSE_COLUMNS + ["output_index"]


def arc_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row arc features; computed once per distinct arc then broadcast."""
    key = ["logic_function", "related_pin", "when", "timing_sense", "output_pin", "cell_base"]
    arcs = df[key].drop_duplicates().reset_index(drop=True)
    pin = np.array([pin_structure(f, p) for f, p in zip(arcs["logic_function"],
                                                         arcs["related_pin"])])
    when = np.array([when_slots(f, p, w) for f, p, w in zip(arcs["logic_function"],
                                                            arcs["related_pin"], arcs["when"])])
    out = pd.DataFrame(np.hstack([pin, when]), columns=PIN_COLUMNS + WHEN_COLUMNS)
    unknown = set(arcs["timing_sense"]) - set(TIMING_SENSES)
    if unknown:
        raise ValueError(f"unknown timing_sense: {sorted(unknown)}")
    for sense, column in zip(TIMING_SENSES, SENSE_COLUMNS):
        out[column] = (arcs["timing_sense"] == sense).astype(float).to_numpy()
    # Multi-output cells (FA/HA: CON, SN): index of the output in sorted pin order.
    outputs = df.groupby("cell_base")["output_pin"].agg(lambda s: sorted(set(s)))
    out["output_index"] = [float(outputs[c].index(o))
                           for c, o in zip(arcs["cell_base"], arcs["output_pin"])]
    arcs = pd.concat([arcs, out], axis=1)
    merged = df[key].merge(arcs, on=key, how="left", validate="many_to_one")
    return merged[ARC_FEATURE_COLUMNS].set_axis(df.index).astype(np.float32)
