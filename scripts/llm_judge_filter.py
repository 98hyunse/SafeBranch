"""C-2 axis-level LLM judge filter.

Drop rule (per branch_id):
  A == 1  OR  B < 3  OR  C < 3   →  drop

Rationale: the sum<12 rule is too aggressive — it bundles "catastrophic"
rows (A=1: chosen failed to resolve the risk) with "evenly mediocre" rows
(A=B=C=3-4). Axis-level rule catches the former exactly without sacrificing
the latter. See data_pipeline_risks_report.html §Risk 2 for the
single-fingerprint comparison (sum<12 drops 31 incl. 25 evenly-mediocre;
axis drops 6 all catastrophic).

Usage:
  python -m scripts.llm_judge_filter \\
    --branches pairs_merged_v5/branches.jsonl \\
    --judges /tmp/judge_results_v5.jsonl \\
    --output pairs_merged_v5_filtered/
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

# Temp-pool provenance suffix is `__t<digits>` (e.g. __t0, __t03, __t07,
# __t10). A bare `"__t" in bid` substring test misfires on task names like
# `clean_a_box_fan__toggled_on_...` or `..._cabinet__with__bag__of__tea`,
# which carry an unrelated `__t`. Require a digit right after `__t`.
_POOL_SUFFIX_RE = re.compile(r"__t\d")


def axis_kill(A: int, B: int, C: int) -> bool:
    return A == 1 or B < 3 or C < 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", required=True, help="merged branches.jsonl")
    ap.add_argument("--judges", required=True, help="judge results jsonl")
    ap.add_argument("--output", required=True, help="output dir")
    ap.add_argument("--strict", action="store_true",
                    help="error if any branch has no matching judge result")
    args = ap.parse_args()

    # Judge branch_ids may carry a trailing `__<pool_tag>` suffix used for
    # uniqueness across launches. New merged rows don't carry that suffix —
    # we strip it and key by (base_id, pool_tag) so we can match against the
    # row's winner_source.
    # v2 judge emits a stable `uid` = source_dir|step_index|branch_id, which
    # is collision-free even when branch_id repeats across step/temp. Prefer
    # uid; fall back to branch_id for legacy v1 judge files.
    def _row_uid(rr: dict) -> str:
        return "|".join(str(rr.get(k, "")) for k in
                        ("source_dir", "step_index", "branch_id"))

    judges_by_uid: dict[str, dict] = {}
    judges_by_base: dict[str, list[dict]] = {}
    has_uid = False
    for line in open(args.judges):
        r = json.loads(line)
        uid = r.get("uid")
        if uid:
            has_uid = True
            judges_by_uid[uid] = r
        bid = r.get("branch_id") or ""
        if bid:
            judges_by_base.setdefault(bid, []).append(r)
    print(f"judges loaded: {sum(len(v) for v in judges_by_base.values())} "
          f"(uid-keyed: {has_uid}, {len(judges_by_uid)} uids / "
          f"{len(judges_by_base)} base_ids)")

    def lookup_judge(row: dict) -> dict | None:
        if has_uid:
            return judges_by_uid.get(_row_uid(row))
        # legacy v1 path: branch_id only (may be 1:N — first match)
        cand = judges_by_base.get(row.get("branch_id") or "", [])
        return cand[0] if cand else None

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    kept, dropped, no_judge = 0, 0, 0
    drop_records = []
    no_judge_ids = []

    with (out / "branches.jsonl").open("w") as fout_keep, \
         (out / "branches_dropped.jsonl").open("w") as fout_drop:
        for line in open(args.branches):
            r = json.loads(line)
            bid = r.get("branch_id")
            j = lookup_judge(r) if bid else None
            if not j or not j.get("ok"):
                no_judge += 1
                no_judge_ids.append(bid)
                # No judge result yet — keep by default but mark for re-judge.
                r["_pending_judge"] = True
                fout_keep.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
                continue
            A, B, C = j.get("A", 0), j.get("B", 0), j.get("C", 0)
            if axis_kill(A, B, C):
                dropped += 1
                drop_records.append({
                    "branch_id": bid, "task": r.get("task"),
                    "step_index": r.get("step_index"),
                    "A": A, "B": B, "C": C,
                    "rationale": j.get("rationale", ""),
                })
                fout_drop.write(json.dumps({**r, "_axis_drop": {"A": A, "B": B, "C": C, "rationale": j.get("rationale", "")}}, ensure_ascii=False) + "\n")
            else:
                fout_keep.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1

    stats = {
        "total_input": kept + dropped,
        "kept": kept,
        "dropped": dropped,
        "no_judge_match": no_judge,
        "drop_rule": "A==1 OR B<3 OR C<3",
        "drop_records_sample": drop_records[:20],
        "no_judge_ids_sample": no_judge_ids[:20],
    }
    with (out / "filter_stats.json").open("w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n=== axis filter ===")
    print(f"input:           {kept + dropped}")
    print(f"kept:            {kept}")
    print(f"dropped:         {dropped}")
    print(f"no_judge_match:  {no_judge}  (kept with _pending_judge flag)")
    print(f"output: {out / 'branches.jsonl'}")
    print(f"dropped: {out / 'branches_dropped.jsonl'}")
    print(f"stats:   {out / 'filter_stats.json'}")

    if args.strict and no_judge > 0:
        raise SystemExit(f"--strict: {no_judge} rows had no matching judge result")


if __name__ == "__main__":
    main()
