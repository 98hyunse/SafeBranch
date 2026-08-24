"""Merge multiple branch pools (per-temp) into one dedupe + provenance pool.

Input: list of pairs_<tag>/ directories produced by run_pipeline.py
Output: pairs_merged/ with dedupe rows, provenance metadata, sft/dpo/review/stats.

Dedupe key = full-fingerprint over (task, risk_type, step_index,
rejected.action, chosen.action, chosen.reasoning, rejected.reasoning).
Two rows are merged only when *every* salient field matches — temperature
variation that produces the same action under different reasoning is kept as
diverse training signal, not collapsed. When the literal duplicate case does
occur and multiple rows share a key, prefer clean over review, source_priority
(t00<t07<t10), then longer chosen.reasoning, then alphabetical.

Review row handling: rows with `validation.needs_review=True` (typically
`same_rule_re_triggered` with trigger_resolved=True) are *valid imitation
targets* but their preference pair is noisy. sft.jsonl includes clean+review;
dpo.jsonl includes clean only. review.jsonl mirrors the review subset for
human inspection.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def norm_action(s: str) -> str:
    a = (s or "").strip().lower().replace(" ", "")
    if a in ("done", "done()"):
        return "done()"
    return a


def dedupe_key(row: dict) -> tuple:
    """Full-fingerprint key: rows are merged only when *every* salient field
    matches. The intent is to preserve diversity from temperature variation —
    if either chosen action, chosen reasoning, or rejected reasoning differs
    by even one character, the rows are kept as distinct training pairs.
    Generalization (same action under varied reasoning) is treated as signal,
    not noise. Only literal duplicates are collapsed.
    """
    chosen = row.get("chosen") or {}
    rejected = row.get("rejected") or {}
    risk_type = (row.get("rule_meta") or {}).get("risk_type", "") \
        or (row.get("critic_feedback") or {}).get("risk_type", "")
    return (
        row.get("task", ""),
        risk_type,
        row.get("step_index"),
        norm_action(rejected.get("action", "")),
        norm_action(chosen.get("action", "")),
        (chosen.get("reasoning") or "").strip(),
        (rejected.get("reasoning") or "").strip(),
    )


def is_review(row: dict) -> bool:
    return bool((row.get("validation") or {}).get("needs_review", False))


def chosen_text_len(row: dict) -> int:
    c = row.get("chosen") or {}
    reasoning = c.get("reasoning") or c.get("rationale") or ""
    return len(str(reasoning)) + len(str(c.get("action", "")))


# Source priority for tiebreaker: lower temperature → more deterministic →
# more typical/stable behavior → preferred. Unknown tags rank last.
SOURCE_PRIORITY = {
    "t00": 0, "t07": 1, "t10": 2,
    "t0_c3": 0, "t03_c3": 1, "t07_c3": 2, "t10_c3": 3,
}


def source_rank(tag: str) -> int:
    return SOURCE_PRIORITY.get(tag, 99)


def full_prompt(row: dict) -> str:
    """Reconstruct the full actor prompt that should match the eval-time form.

    task_input already contains the `- history_actions: …` line, so do *not*
    re-append `prompt.history_actions` separately (that field is preserved on
    the row for analysis/debugging, but appending it here duplicates the history
    section and breaks training/eval form parity).
    """
    p = row.get("prompt") or {}
    if isinstance(p, str):
        return p
    parts = [p.get("common", ""), p.get("task_input", "")]
    return "\n\n".join(s for s in parts if s)


def pick_winner(tagged_rows: list[tuple[str, dict]]) -> tuple[str, dict, str]:
    """Among rows sharing the same key, pick the canonical representative.

    Quality stack (top → bottom):
      1. clean > review                                   (`clean_vs_review`)
      2. source_priority: t00 < t07 < t10                 (`source_priority`)
      3. longer chosen text                               (`length`)
      4. alphabetical branch_id                           (`branch_id`)

    Returns (winner_tag, winner_row, tiebreak_dimension) — `tiebreak_dimension`
    names the first stack level on which the winner outranks the runner-up,
    so we can measure which signal is actually doing work.
    """
    def sort_key(item):
        tag, r = item
        return (
            1 if is_review(r) else 0,
            source_rank(tag),
            -chosen_text_len(r),
            r.get("branch_id", ""),
        )

    ordered = sorted(tagged_rows, key=sort_key)
    win_tag, win_row = ordered[0]
    if len(ordered) == 1:
        return win_tag, win_row, "uncontested"

    runner_tag, runner_row = ordered[1]
    if is_review(win_row) != is_review(runner_row):
        dim = "clean_vs_review"
    elif source_rank(win_tag) != source_rank(runner_tag):
        dim = "source_priority"
    elif chosen_text_len(win_row) != chosen_text_len(runner_row):
        dim = "length"
    else:
        dim = "branch_id"
    return win_tag, win_row, dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, nargs="+",
                    help="Input pool dirs (each contains branches.jsonl). Format: <tag>=<path>")
    ap.add_argument("--output", required=True, help="Output dir for merged pool.")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load every input pool, tagged with its source label.
    by_key: dict[tuple, list[tuple[str, dict]]] = defaultdict(list)
    raw_total = 0
    per_tag_count = {}
    for spec in args.inputs:
        if "=" not in spec:
            raise ValueError(f"Input spec must be <tag>=<path>, got: {spec}")
        tag, path = spec.split("=", 1)
        bjsonl = Path(path) / "branches.jsonl"
        rows = [json.loads(l) for l in open(bjsonl)]
        per_tag_count[tag] = len(rows)
        for r in rows:
            by_key[dedupe_key(r)].append((tag, r))
            raw_total += 1
    print(f"loaded {raw_total} rows across {len(args.inputs)} pools: {per_tag_count}")
    print(f"unique keys: {len(by_key)}")

    # Merge: pick winner per key, attach provenance.
    merged_rows = []
    tiebreak_dim_counter = Counter()
    for key, tagged_rows in by_key.items():
        sources = sorted({tag for tag, _ in tagged_rows})
        winner_tag, winner_row, tiebreak_dim = pick_winner(tagged_rows)
        if len(tagged_rows) >= 2:
            tiebreak_dim_counter[tiebreak_dim] += 1
        dropped_branch_ids = [
            {"tag": tag, "branch_id": r.get("branch_id")}
            for tag, r in tagged_rows if r is not winner_row
        ]
        winner = dict(winner_row)  # don't mutate input
        winner["temp_sources"] = sources
        winner["source_count"] = len(sources)
        winner["dedupe_dropped_count"] = len(tagged_rows) - 1
        winner["winner_source"] = winner_tag
        winner["tiebreak_dimension"] = tiebreak_dim
        winner["dropped_branch_ids"] = dropped_branch_ids
        merged_rows.append(winner)

    clean_rows = [r for r in merged_rows if not is_review(r)]
    review_rows = [r for r in merged_rows if is_review(r)]
    # 2026-05-15: review rows kept in both SFT and DPO — `same_rule_re_triggered`
    # is mostly a triggering_violation[0] mis-classification artifact (see
    # project_term_safety_review_false_alarm), not real noise. LLM judge axis
    # filter (A=1 OR B<3 OR C<3) removes any genuinely bad rows downstream.
    sft_rows = clean_rows + review_rows
    dpo_rows = clean_rows + review_rows
    print(
        f"merged: {len(merged_rows)} rows "
        f"({len(clean_rows)} clean / {len(review_rows)} review) — "
        f"sft={len(sft_rows)} dpo={len(dpo_rows)} (both clean+review)"
    )

    # Schema validation: every row must carry chosen + rejected for downstream use.
    for r in merged_rows:
        missing = [k for k in ("chosen", "rejected") if k not in r]
        if missing:
            raise ValueError(
                f"Row missing required fields {missing}: "
                f"task={r.get('task')} step={r.get('step_index')} "
                f"branch_id={r.get('branch_id')} pool={r.get('winner_source')}"
            )

    # Write branches.jsonl (full, with provenance).
    with (out / "branches.jsonl").open("w") as f:
        for r in merged_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write sft.jsonl (chosen only) — clean + review.
    # Carries all fields needed by downstream training-data conversion.
    # need so they do not have to re-read branches.jsonl and re-apply policy.
    with (out / "sft.jsonl").open("w") as f:
        for r in sft_rows:
            sft = {
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
                "temp_sources": r["temp_sources"],
                "source_count": r["source_count"],
                "needs_review": is_review(r),
            }
            f.write(json.dumps(sft, ensure_ascii=False) + "\n")

    # Write dpo.jsonl (preference pairs) — clean only.
    with (out / "dpo.jsonl").open("w") as f:
        for r in dpo_rows:
            dpo = {
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
                "temp_sources": r["temp_sources"],
                "source_count": r["source_count"],
            }
            f.write(json.dumps(dpo, ensure_ascii=False) + "\n")

    # Write review.jsonl.
    with (out / "review.jsonl").open("w") as f:
        for r in review_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats.
    by_risk = Counter()
    by_source_kind = Counter()
    by_source_count = Counter()
    by_temp_sources = Counter()
    by_task = defaultdict(Counter)
    null_risk_type_count = 0
    for r in merged_rows:
        rt = (r.get("rule_meta") or {}).get("risk_type", "") \
            or (r.get("critic_feedback") or {}).get("risk_type", "")
        if rt:
            by_risk[rt] += 1
        else:
            null_risk_type_count += 1
        by_source_kind[r.get("source_kind", "?")] += 1
        by_source_count[r["source_count"]] += 1
        by_temp_sources[",".join(r["temp_sources"])] += 1
        by_task[r["task"]][r.get("source_kind", "?")] += 1

    # tiebreak dimension distribution (only contested groups, i.e. ≥2 candidates):
    # which stack level (clean_vs_review / source_priority / length / branch_id)
    # actually decided the winner. `length` ratio is the headline metric for
    # whether the new quality stack is doing real work vs. the old length-only.
    contested_groups = sum(1 for tagged in by_key.values() if len(tagged) >= 2)
    tiebreak_used_count = sum(
        v for k, v in tiebreak_dim_counter.items() if k != "uncontested"
    )

    stats = {
        "raw_total": raw_total,
        "raw_per_tag": per_tag_count,
        "merged_total": len(merged_rows),
        "retention_rate": round(len(merged_rows) / raw_total, 4) if raw_total else 0,
        "clean": len(clean_rows),
        "review": len(review_rows),
        "sft_row_count": len(sft_rows),
        "dpo_row_count": len(dpo_rows),
        "review_policy": "sft_and_dpo_both_include_review",
        "review_in_sft": len(review_rows),
        "review_in_dpo": len(review_rows),
        "contested_groups": contested_groups,
        "tiebreak_used_count": tiebreak_used_count,
        "tiebreak_dimension_distribution": dict(tiebreak_dim_counter),
        "null_risk_type_count": null_risk_type_count,
        "winner_source_distribution": dict(Counter(r.get("winner_source") for r in merged_rows)),
        "by_risk_type": dict(by_risk),
        "by_source_kind": dict(by_source_kind),
        "by_source_count": dict(by_source_count),
        "by_temp_sources": dict(by_temp_sources),
        "by_task_top20": dict(sorted(
            by_task.items(),
            key=lambda x: -sum(x[1].values()),
        )[:20]),
    }
    with (out / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"done — {out}")
    print(json.dumps({k: stats[k] for k in [
        "raw_total", "merged_total", "clean", "review",
        "sft_row_count", "dpo_row_count", "review_policy",
        "by_source_count", "by_temp_sources",
    ]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
