"""Select canonical ASAP7 NLDM sources and emit a pinned provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

PATTERN = re.compile(
    r"asap7sc7p5t_(AO|OA|INVBUF|SIMPLE|SEQ)_(LVT|RVT|SLVT|SRAM)_"
    r"(FF|SS|TT)_nldm_(\d+)\.lib(?:\.gz)?$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(repo: Path) -> dict:
    source_dir = repo / "flow/platforms/asap7/lib/NLDM"
    selected = {}
    for path in source_dir.iterdir():
        match = PATTERN.fullmatch(path.name)
        if not match or "FAKE" in path.name.upper():
            continue
        family, vt, corner, revision = match.groups()
        key = (family, vt, corner)
        if key not in selected or revision > selected[key][0]:
            selected[key] = (revision, path)
    expected = {(family, vt, corner) for family in ("AO", "OA", "INVBUF", "SIMPLE", "SEQ")
                for vt in ("LVT", "RVT", "SLVT", "SRAM") for corner in ("FF", "SS", "TT")}
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        raise RuntimeError(f"ASAP7 canonical coverage incomplete: {missing}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    files = []
    for (family, vt, corner), (revision, path) in sorted(selected.items()):
        files.append({
            "path": path.relative_to(repo).as_posix(), "family": family,
            "vt_class": vt, "corner": corner, "revision": revision,
            "bytes": path.stat().st_size, "sha256": sha256(path),
        })
    return {
        "schema_version": 1,
        "repository": "https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git",
        "commit": commit,
        "selection_rule": "latest non-FAKE AO/OA/INVBUF/SIMPLE/SEQ for every LVT/RVT/SLVT/SRAM x FF/SS/TT",
        "excluded": ["*_FAKE.lib", "fakeram*.lib", "fakeregfile*.lib", "older duplicate revisions"],
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"selected={len(manifest['files'])} commit={manifest['commit']} out={args.out}")


if __name__ == "__main__":
    main()
