"""PLAN v4 W1: cell_base -> (family, topology type, size, VT) catalog with review flags."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from plan_v4_data import MAINLINE_FAMILIES, VT_ORDER, parse_cell_base

# Topology types whose merge decision is not derivable from the name alone.
# Each entry: type stem regex -> (reason, proposal). Proposals are defaults the
# user must confirm; the catalog never merges them silently.
REVIEW_RULES = (
    (r"^BUFx\d+f$", "名稱在 drive 後多 'f'；同 drive 的 pin cap 約為 BUF 的 2 倍、delay 較低，拓樸可能不同",
     "keep_separate"),
    (r"^HB\d$", "HB1–HB4 皆為 xp67，數字是 stage／delay 等級而非 drive；delay 隨編號遞增",
     "keep_separate"),
    (r"^CKINVDC$", "與 INV 同為 !A，但為 clock 專用 inverter（DC 結構）",
     "keep_separate"),
)


def build_catalog(cells: pd.DataFrame, table_counts: pd.DataFrame) -> pd.DataFrame:
    """`cells`: unique (cell_base, family, logic_class, vt_class); `table_counts`: per cell_base."""
    bases = cells.groupby("cell_base").agg(
        family=("family", "first"), logic_class=("logic_class", "first"),
        n_family=("family", "nunique"), n_logic=("logic_class", "nunique"),
        vts=("vt_class", lambda s: "|".join(v for v in VT_ORDER if v in set(s))),
        n_vt=("vt_class", "nunique"),
    ).reset_index()
    if (bases[["n_family", "n_logic"]] > 1).any().any():
        raise ValueError("cell_base maps to more than one family or logic class")
    parsed = bases["cell_base"].map(parse_cell_base)
    bases["topology_type"] = parsed.map(lambda p: p[0])
    bases["size_label"] = parsed.map(lambda p: p[1])
    bases["size"] = parsed.map(lambda p: p[2])
    bases["name_suffix"] = parsed.map(lambda p: p[3])
    bases["in_mainline"] = bases["family"].isin(MAINLINE_FAMILIES)
    bases = bases.merge(table_counts, on="cell_base", how="left")
    bases["delay_tables"] = bases["delay_tables"].fillna(0).astype(int)

    per_type = bases.groupby("topology_type").agg(
        n_sizes_in_type=("size", "nunique"), type_families=("family", "nunique"))
    if (per_type["type_families"] > 1).any():
        bad = per_type.index[per_type["type_families"] > 1].tolist()
        raise ValueError(f"topology types spanning families: {bad}")
    bases = bases.merge(per_type[["n_sizes_in_type"]], left_on="topology_type", right_index=True)
    bases["size_rank_in_type"] = bases.groupby("topology_type")["size"].rank(method="dense")

    reasons, proposals = [], []
    for row in bases.itertuples():
        reason, proposal = "", "auto"
        for pattern, why, default in REVIEW_RULES:
            if re.match(pattern, row.topology_type):
                reason, proposal = why, default
        if not reason and row.size_label is None:
            reason, proposal = "名稱無 drive-strength token", "keep_separate"
        if not reason and row.n_vt != len(VT_ORDER):
            reason, proposal = f"VT 不齊：{row.vts}", "keep_separate"
        reasons.append(reason)
        proposals.append(proposal)
    bases["needs_review"] = [bool(r) for r in reasons]
    bases["review_reason"] = reasons
    bases["merge_proposal"] = proposals
    columns = ["cell_base", "family", "in_mainline", "logic_class", "topology_type",
               "size_label", "size", "name_suffix", "n_sizes_in_type", "size_rank_in_type",
               "vts", "delay_tables", "needs_review", "review_reason", "merge_proposal"]
    return bases[columns].sort_values(["family", "topology_type", "size"]).reset_index(drop=True)


def summarize(catalog: pd.DataFrame) -> dict:
    main = catalog[catalog["in_mainline"]]
    types = main.groupby("topology_type")["n_sizes_in_type"].first()
    by_family = main.groupby("family").agg(
        cell_base=("cell_base", "size"), function_types=("topology_type", "nunique"),
        delay_tables=("delay_tables", "sum"))
    multi = main.drop_duplicates("topology_type").groupby("family")["n_sizes_in_type"].apply(
        lambda s: int((s > 1).sum()))
    by_family["multi_size_types"] = multi
    return {
        "mainline_cell_base": int(len(main)),
        "mainline_function_types": int(types.size),
        "mainline_multi_size_types": int((types > 1).sum()),
        "mainline_single_size_types": int((types == 1).sum()),
        "mainline_delay_tables": int(main["delay_tables"].sum()),
        "excluded_seq_cell_base": int((~catalog["in_mainline"]).sum()),
        "needs_review_cell_base": catalog.loc[catalog["needs_review"], "cell_base"].tolist(),
        "by_family": {
            fam: {k: int(v) for k, v in row.items()} for fam, row in by_family.iterrows()
        },
    }


def spot_check_sample(catalog: pd.DataFrame, n: int = 10, seed: int = 20261018) -> pd.DataFrame:
    """Deterministic sample for manual check: one per mainline family first, then random."""
    main = catalog[catalog["in_mainline"]].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    picked = [int(rng.choice(np.flatnonzero(main["family"] == fam)))
              for fam in MAINLINE_FAMILIES if (main["family"] == fam).any()]
    rest = np.setdiff1d(np.arange(len(main)), picked)
    picked += rng.choice(rest, size=max(0, n - len(picked)), replace=False).tolist()
    return main.iloc[sorted(picked)].reset_index(drop=True)
