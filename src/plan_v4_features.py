"""PLAN v4 W2: full per-row feature table (section 4.3) for L1/L2 training.

The cache holds every candidate column; each experiment selects its columns
from the manifest (e.g. E2 uses `vt_ordinal` and `pin_cap_ref_vt`, never the
held-out VT's own pin cap). Column groups are listed so selection is explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arc_features import ARC_FEATURE_COLUMNS, arc_feature_frame
from features import build_feature_frame
from plan_v4_data import CORNER_ORDER, VT_ORDER

ORDINAL_COLUMNS = ["vt_ordinal", "corner_ordinal", "is_rise"]
PIN_CAP_COLUMNS = ["pin_cap_own", "pin_cap_ref_vt", "pin_cap_ref_vt_lvt", "pin_cap_ref_tt"]


def reference_pin_caps(df: pd.DataFrame) -> pd.DataFrame:
    """Own pin cap plus the reference values PLAN 4.3 allows for E2 (RVT/LVT) and E4 (TT)."""
    key = ["cell_base", "related_pin"]
    caps = df[key + ["vt_class", "corner", "related_pin_cap_ff"]].drop_duplicates(
        key + ["vt_class", "corner"])

    def ref(column: str, value: str, name: str) -> pd.Series:
        other = [c for c in ("vt_class", "corner") if c != column]
        table = caps[caps[column] == value][key + other + ["related_pin_cap_ff"]]
        table = table.rename(columns={"related_pin_cap_ff": name})
        merged = df[key + other].merge(table, on=key + other, how="left", validate="many_to_one")
        return merged[name].set_axis(df.index)

    out = pd.DataFrame(index=df.index)
    out["pin_cap_own"] = df["related_pin_cap_ff"].astype(float)
    out["pin_cap_ref_vt"] = ref("vt_class", "RVT", "pin_cap_ref_vt")
    out["pin_cap_ref_vt_lvt"] = ref("vt_class", "LVT", "pin_cap_ref_vt_lvt")
    out["pin_cap_ref_tt"] = ref("corner", "TT", "pin_cap_ref_tt")
    if out.isna().any().any():
        raise ValueError("reference pin cap missing for some rows")
    return out


def build_v4_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    base = build_feature_frame(df)
    arcs = arc_feature_frame(df)
    extra = pd.DataFrame(index=df.index)
    extra["vt_ordinal"] = df["vt_class"].map({v: i for i, v in enumerate(VT_ORDER)})
    extra["corner_ordinal"] = df["corner"].map({c: i for i, c in enumerate(CORNER_ORDER)})
    extra["is_rise"] = (df["table_type"] == "cell_rise").astype(float)
    if extra.isna().any().any():
        raise ValueError("unknown VT or corner label")
    caps = reference_pin_caps(df)
    out = pd.concat([base, arcs, extra.astype(np.float32), caps.astype(np.float32)], axis=1)
    return out.astype(np.float32)


def column_groups(columns: list[str]) -> dict[str, list[str]]:
    vt_onehot = [c for c in columns if c.startswith("vt_") and c != "vt_ordinal"]
    return {
        "arc": list(ARC_FEATURE_COLUMNS),
        "ordinal": list(ORDINAL_COLUMNS),
        "pin_cap": list(PIN_CAP_COLUMNS),
        "vt_onehot": vt_onehot,
        "base_other": [c for c in columns if c not in set(ARC_FEATURE_COLUMNS)
                       | set(ORDINAL_COLUMNS) | set(PIN_CAP_COLUMNS) | set(vt_onehot)],
    }
