"""PLAN v4 W1: rows whose current feature vector maps to more than one delay target.

A conflict means two rows share every value of `features.build_feature_frame`
(the 8/15 feature set) yet carry different golden delays, so no function of
those features can fit both. The report then adds the candidate columns from
PLAN v4 section 4.3 one at a time to show which ones remove the conflict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import build_feature_frame

CANDIDATE_COLUMNS = ("when", "related_pin", "timing_sense", "output_pin")


def _conflict_mask(keys: pd.Series, target: pd.Series) -> np.ndarray:
    frame = pd.DataFrame({"k": keys.to_numpy(), "v": target.to_numpy()})
    return (frame.groupby("k")["v"].transform("nunique") > 1).to_numpy()


def _row_hash(frame: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(frame, index=False)


def conflict_report(df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """`df`: mainline delay rows with the columns `build_feature_frame` needs.

    Returns the summary and the per-row baseline conflict mask.
    """
    base_hash = _row_hash(build_feature_frame(df))
    base = _conflict_mask(base_hash, df["value"])
    report = {
        "rows": int(len(df)),
        "baseline_conflict_rows": int(base.sum()),
        "baseline_conflict_ratio": float(base.mean()),
        "baseline_distinct_feature_vectors": int(base_hash.nunique()),
    }
    added = {}
    for column in CANDIDATE_COLUMNS:
        keys = _row_hash(pd.DataFrame({"h": base_hash, "c": df[column].astype(str)}))
        added[column] = int(_conflict_mask(keys, df["value"]).sum())
    report["conflict_rows_if_add_one"] = added
    cumulative, frame = {}, pd.DataFrame({"h": base_hash})
    for column in CANDIDATE_COLUMNS:
        frame[column] = df[column].astype(str).to_numpy()
        cumulative["+".join(frame.columns[1:])] = int(
            _conflict_mask(_row_hash(frame), df["value"]).sum())
    report["conflict_rows_if_add_cumulative"] = cumulative

    flagged = df.assign(conflict=base, has_when=df["when"].astype(str) != "")
    report["by_family"] = {
        fam: {"rows": int(len(g)), "conflict_rows": int(g["conflict"].sum()),
              "conflict_ratio": float(g["conflict"].mean())}
        for fam, g in flagged.groupby("family")
    }
    report["by_has_when"] = {
        str(key): {"rows": int(len(g)), "conflict_rows": int(g["conflict"].sum()),
                   "conflict_ratio": float(g["conflict"].mean())}
        for key, g in flagged.groupby("has_when")
    }
    return report, base


def top_conflict_types(df: pd.DataFrame, conflict: np.ndarray, n: int = 15) -> pd.DataFrame:
    flagged = df.assign(conflict=conflict)
    out = flagged.groupby(["family", "topology_type"]).agg(
        rows=("value", "size"), conflict_rows=("conflict", "sum"))
    out["conflict_ratio"] = out["conflict_rows"] / out["rows"]
    return out.sort_values("conflict_rows", ascending=False).head(n).reset_index()
