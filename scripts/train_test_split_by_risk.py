"""Task-level train/test split stratified by primary risk_type.

Each task is assigned a primary risk_type (the one that appears most often
across its pairs). Tasks within each risk_type are then split 70/30, so
*every* risk_type appears in both train and test. This measures whether
the model can transfer a safety principle to new objects/locations within
the same principle (e.g. "Collision Hazard learned on cabinet placement
→ tested on shoe rack placement").

Multi-label note: 40% of tasks carry multiple risk_types. We split by the
*primary* (most frequent) one — minor risks ride along with their host
task. This keeps the per-risk-type test count stable and the manifest
unambiguous.

Output: pair_splits_by_risk/
  train.jsonl
  test.jsonl
  split_manifest.json
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path


TEST_FRACTION = 0.30
SEED = 20260515


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="filtered branches.jsonl (681 pair after axis filter)")
    ap.add_argument("--output", required=True, help="output dir")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.input)]
    print(f"loaded {len(rows)} pairs")

    # Per-task primary risk_type.
    task_risk_counts: dict[str, Counter] = defaultdict(Counter)
    task_rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        t = r.get("task", "")
        rt = (r.get("rule_meta") or {}).get("risk_type", "") \
            or (r.get("critic_feedback") or {}).get("risk_type", "")
        if rt:
            task_risk_counts[t][rt] += 1
        task_rows[t].append(r)

    task_primary_risk: dict[str, str] = {}
    for t, counts in task_risk_counts.items():
        if counts:
            task_primary_risk[t] = counts.most_common(1)[0][0]
        else:
            task_primary_risk[t] = "Unknown"

    # All tasks that produced pairs but have no risk_type info get bucketed
    # under "Unknown" — they're still real training data, just not stratifiable.
    for t in task_rows:
        if t not in task_primary_risk:
            task_primary_risk[t] = "Unknown"

    # Group tasks by primary risk.
    risk_to_tasks: dict[str, list[str]] = defaultdict(list)
    for t, rt in task_primary_risk.items():
        risk_to_tasks[rt].append(t)
    for rt in risk_to_tasks:
        risk_to_tasks[rt].sort()

    print(f"tasks: {len(task_primary_risk)} across {len(risk_to_tasks)} risk_types")

    # Stratified split.
    rng = random.Random(args.seed)
    train_tasks: list[str] = []
    test_tasks: list[str] = []
    per_risk_split: dict[str, dict] = {}
    for rt in sorted(risk_to_tasks):
        tasks = risk_to_tasks[rt][:]
        rng.shuffle(tasks)
        n_test = max(1, round(len(tasks) * TEST_FRACTION)) if len(tasks) >= 2 else 0
        test = tasks[:n_test]
        train = tasks[n_test:]
        train_tasks.extend(train)
        test_tasks.extend(test)
        per_risk_split[rt] = {
            "total": len(tasks),
            "train": len(train),
            "test": len(test),
        }

    train_set = set(train_tasks)
    test_set = set(test_tasks)
    assert train_set.isdisjoint(test_set)

    train_pairs = [r for r in rows if r.get("task") in train_set]
    test_pairs = [r for r in rows if r.get("task") in test_set]

    def write_jsonl(path, items):
        with open(path, "w") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(out / "train.jsonl", train_pairs)
    write_jsonl(out / "test.jsonl", test_pairs)

    manifest = {
        "seed": args.seed,
        "test_fraction": TEST_FRACTION,
        "split_strategy": "risk_type stratified (primary risk_type per task)",
        "per_risk_split": per_risk_split,
        "task_counts": {
            "train": len(train_tasks),
            "test": len(test_tasks),
            "total": len(train_tasks) + len(test_tasks),
        },
        "pair_counts": {
            "train": len(train_pairs),
            "test": len(test_pairs),
            "total": len(train_pairs) + len(test_pairs),
        },
        "task_primary_risk": task_primary_risk,
        "task_lists": {
            "train": sorted(train_tasks),
            "test": sorted(test_tasks),
        },
    }
    with (out / "split_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n=== risk-stratified split (seed={args.seed}) ===")
    print(f"{'risk_type':<25s} {'total':>6s} {'train':>7s} {'test':>6s}")
    print("-" * 50)
    for rt in sorted(per_risk_split, key=lambda x: -per_risk_split[x]["total"]):
        s = per_risk_split[rt]
        print(f"{rt:<25s} {s['total']:>6d} {s['train']:>7d} {s['test']:>6d}")
    print("-" * 50)
    print(f"{'TOTAL':<25s} {manifest['task_counts']['total']:>6d} "
          f"{manifest['task_counts']['train']:>7d} {manifest['task_counts']['test']:>6d}")
    print(f"\nPair counts:")
    print(f"  train: {len(train_pairs)}")
    print(f"  test:  {len(test_pairs)}")
    print(f"\noutput: {out}")


if __name__ == "__main__":
    main()
