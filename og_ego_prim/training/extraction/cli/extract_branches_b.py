"""CLI: collect Track B branch_*.json files into branches.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from og_ego_prim.training.extraction.branch_extractor_track_b import (
    collect_branches_from_results_dir,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    rows = collect_branches_from_results_dir(Path(args.results_dir))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "branches.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"collected {len(rows)} rows → {out / 'branches.jsonl'}")


if __name__ == "__main__":
    main()
