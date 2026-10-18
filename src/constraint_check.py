"""PLAN v4 section 6.1: golden compliance of physical-consistency rules on mainline delay.

Every rule is a comparison between golden delay values; the compliance rate is
`ok / compared` with the comparison unit stated per rule, so the same function
can later score model predictions by passing a predicted `value` column.
Ties (equal values) count as compliant for the non-strict orderings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plan_v4_data import ARC_COLUMNS, CORNER_ORDER, TABLE_KEY, VT_ORDER

HARD_THRESHOLD = 0.999
PENALTY_THRESHOLD = 0.95  # PLAN 6.1: slew (98.4%) and VT (99.6%) are penalties; size (~79%) is not


def tables_as_cube(df: pd.DataFrame, value: str = "value") -> tuple[pd.DataFrame, np.ndarray]:
    """Reshape rows into (tables, slew, load) assuming each table is a full rectangular grid."""
    ordered = df.sort_values(TABLE_KEY + ["slew", "load"], kind="mergesort")
    sizes = ordered.groupby(TABLE_KEY, sort=False, dropna=False).size()
    n_slew = ordered.groupby(TABLE_KEY, sort=False, dropna=False)["slew"].nunique()
    n_load = ordered.groupby(TABLE_KEY, sort=False, dropna=False)["load"].nunique()
    shapes = set(zip(n_slew, n_load))
    if len(shapes) != 1 or not (sizes == n_slew * n_load).all():
        raise ValueError(f"tables are not full grids of one shape: {sorted(shapes)[:5]}")
    (ns, nl), = shapes
    keys = ordered[TABLE_KEY].iloc[:: ns * nl].reset_index(drop=True)
    return keys, ordered[value].to_numpy(dtype=float).reshape(-1, ns, nl)


def _rate(ok: np.ndarray) -> dict:
    ok = np.asarray(ok, dtype=bool).ravel()
    return {"compared": int(ok.size), "ok": int(ok.sum()),
            "rate": float(ok.mean()) if ok.size else float("nan")}


def _align(keys: pd.DataFrame, by: list[str], axis: str, levels: tuple[str, ...]) -> np.ndarray:
    """Table indices with one column per level of `axis`; rows missing a level are dropped."""
    # Use the cube positions carried in the index, so filtered `keys` still point correctly.
    index = keys.assign(_i=keys.index.to_numpy())
    wide = index.pivot_table(index=by, columns=axis, values="_i", aggfunc="first")
    return wide.reindex(columns=list(levels)).dropna().astype(int).to_numpy()


def rule_load_monotonic(cube: np.ndarray) -> dict:
    return {"unit": "adjacent load pairs", **_rate(np.diff(cube, axis=2) >= 0)}


def rule_slew_monotonic(cube: np.ndarray) -> dict:
    ok = np.diff(cube, axis=1) >= 0
    out = {"unit": "adjacent slew pairs", **_rate(ok)}
    out["tables_fully_ok"] = float(ok.all(axis=(1, 2)).mean())
    bad = ~ok
    out["violations_by_slew_step"] = bad.sum(axis=(0, 2)).astype(int).tolist()
    return out


def rule_corner_order(keys: pd.DataFrame, cube: np.ndarray) -> dict:
    idx = _align(keys, ["cell_base", "vt_class"] + ARC_COLUMNS, "corner", CORNER_ORDER)
    ff, tt, ss = (cube[idx[:, k]] for k in range(3))
    out = {"unit": "grid points with FF<=TT<=SS", **_rate((ff <= tt) & (tt <= ss))}
    out["table_triples"] = int(len(idx))
    out["ff_le_ss_rate"] = float((ff <= ss).mean())
    return out


def rule_vt_order(keys: pd.DataFrame, cube: np.ndarray, corner: str | None = "TT") -> dict:
    sub = keys if corner is None else keys[keys["corner"] == corner]
    by = ["cell_base", "corner"] + ARC_COLUMNS
    idx = _align(sub, by, "vt_class", VT_ORDER)
    vals = [cube[idx[:, k]] for k in range(len(VT_ORDER))]
    chain = np.logical_and.reduce([vals[k] <= vals[k + 1] for k in range(len(vals) - 1)])
    out = {"unit": f"grid points with SLVT<=LVT<=RVT<=SRAM ({corner or 'all corners'})",
           **_rate(chain)}
    out["table_quads"] = int(len(idx))
    out["adjacent_pair_rate"] = float(np.mean(
        [(vals[k] <= vals[k + 1]).mean() for k in range(len(vals) - 1)]))
    return out


def rule_size_delay(rows: pd.DataFrame) -> dict:
    """Within a topology type: next larger size has delay <= smaller, same arc/slew/load."""
    key = ["topology_type", "vt_class", "corner"] + ARC_COLUMNS + ["slew", "load"]
    ok = _adjacent_size_ok(rows[key + ["size", "value"]], key, "value", larger_is="le")
    return {"unit": "adjacent-size point pairs, same arc/slew/load", **_rate(ok)}


def rule_size_pin_cap(rows: pd.DataFrame) -> dict:
    """Within a topology type: next larger size has related-pin cap >= smaller, same pin."""
    key = ["topology_type", "vt_class", "corner", "related_pin"]
    pins = rows.drop_duplicates(["cell_base"] + key)[key + ["size", "related_pin_cap_ff"]]
    ok = _adjacent_size_ok(pins, key, "related_pin_cap_ff", larger_is="ge")
    return {"unit": "adjacent-size (pin, vt, corner) pairs", **_rate(ok)}


def _adjacent_size_ok(frame: pd.DataFrame, key: list[str], col: str, larger_is: str) -> np.ndarray:
    ordered = frame.sort_values(key + ["size"], kind="mergesort")
    nxt = ordered.groupby(key, sort=False, dropna=False)[col].shift(-1)
    has = nxt.notna().to_numpy()
    cur, nxt = ordered[col].to_numpy()[has], nxt.to_numpy()[has]
    return nxt <= cur if larger_is == "le" else nxt >= cur


def rule_rise_fall_consistency(rows: pd.DataFrame) -> dict:
    """Per (topology type, related pin): share of points agreeing with the majority rise>fall sign."""
    key = ["cell_base", "vt_class", "corner", "output_pin", "related_pin", "timing_sense",
           "timing_type", "when", "slew", "load"]
    wide = rows.pivot_table(index=key + ["topology_type"], columns="table_type",
                            values="value", aggfunc="first").dropna().reset_index()
    rise_gt = (wide["cell_rise"] > wide["cell_fall"]).groupby(
        [wide["topology_type"], wide["related_pin"]]).mean()
    agree = np.maximum(rise_gt, 1 - rise_gt)
    return {"unit": "(topology type, related pin) groups; per-group majority agreement",
            "groups": int(agree.size), "median": float(agree.median()),
            "p10": float(agree.quantile(0.10)), "rate": float(agree.median())}


def recommend(rate: float) -> str:
    if rate >= HARD_THRESHOLD:
        return "hard"
    if rate >= PENALTY_THRESHOLD:
        return "penalty"
    return "report_only"


def check_all(rows: pd.DataFrame) -> dict:
    """`rows`: mainline delay rows with catalog columns `topology_type` and `size` merged in."""
    keys, cube = tables_as_cube(rows)
    results = {
        "load_monotonic": rule_load_monotonic(cube),
        "slew_monotonic": rule_slew_monotonic(cube),
        "corner_order": rule_corner_order(keys, cube),
        "vt_order_tt": rule_vt_order(keys, cube, "TT"),
        "vt_order_all_corners": rule_vt_order(keys, cube, None),
        "size_delay": rule_size_delay(rows),
        "size_pin_cap": rule_size_pin_cap(rows),
        "rise_fall_consistency": rule_rise_fall_consistency(rows),
    }
    for name, res in results.items():
        res["recommendation"] = recommend(res["rate"])
    results["_meta"] = {"tables": int(len(keys)), "grid": list(cube.shape[1:]),
                        "hard_threshold": HARD_THRESHOLD, "penalty_threshold": PENALTY_THRESHOLD}
    return results
