"""Convert a split branches.jsonl (post filter+review+split) into the
sft.jsonl / dpo.jsonl shape that prepare_training_data.py expects.

merge_branch_pools.py emits sft/dpo for the *full* merged pool. After we
filter + user-review + task-level split, only branches.jsonl carries the
final train set, so we re-apply the exact same sft/dpo projection (reusing
merge_branch_pools.full_prompt / is_review) to the split file.

Policy mirrors current merge_branch_pools: both sft and dpo = clean+review
(is_review only annotated via needs_review, not excluded).

Usage:
  python scripts/split_to_sft_dpo.py \
    --input _workspace/pair_splits_by_risk_v5_v2_reviewed/train.jsonl \
    --output_dir _workspace/pair_splits_by_risk_v5_v2_reviewed
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

try:
    from scripts.merge_branch_pools import full_prompt, is_review
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from merge_branch_pools import full_prompt, is_review


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="split branches.jsonl (e.g. train.jsonl)")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_sft = n_dpo = n_review = 0
    with (out / "sft.jsonl").open("w") as fs, (out / "dpo.jsonl").open("w") as fd:
        for r in rows:
            rev = is_review(r)
            n_review += int(rev)
            fs.write(json.dumps({
                "task": r["task"],
                "step_index": r.get("step_index"),
                "recursion_from": r.get("recursion_from"),
                "source_kind": r.get("source_kind"),
                "source_dir": r.get("source_dir"),
                "rule_id": r.get("rule_id"),
                "rule_meta": r.get("rule_meta") or {},
                "branch_id": r.get("branch_id"),
                "prompt": full_prompt(r),
                "completion": r["chosen"],
                "image_paths": r.get("image_paths", []),
                "temp_sources": r.get("temp_sources"),
                "source_count": r.get("source_count"),
                "needs_review": rev,
            }, ensure_ascii=False) + "\n")
            n_sft += 1
            if r.get("rejected"):
                fd.write(json.dumps({
                    "task": r["task"],
                    "step_index": r.get("step_index"),
                    "recursion_from": r.get("recursion_from"),
                    "source_kind": r.get("source_kind"),
                    "source_dir": r.get("source_dir"),
                    "rule_id": r.get("rule_id"),
                    "rule_meta": r.get("rule_meta") or {},
                    "branch_id": r.get("branch_id"),
                    "prompt": full_prompt(r),
                    "chosen": r["chosen"],
                    "rejected": r["rejected"],
                    "critic_feedback": r.get("critic_feedback"),
                    "image_paths": r.get("image_paths", []),
                    "temp_sources": r.get("temp_sources"),
                    "source_count": r.get("source_count"),
                }, ensure_ascii=False) + "\n")
                n_dpo += 1

    print(f"input rows: {len(rows)}")
    print(f"sft.jsonl : {n_sft}  (needs_review={n_review})")
    print(f"dpo.jsonl : {n_dpo}  (rows with rejected)")
    print(f"out: {out}")


if __name__ == "__main__":
    main()
