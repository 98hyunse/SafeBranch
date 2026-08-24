"""Walk _trace.json events to validate TermSafety branches post-hoc.

Schema (new, exp10 + Phase 7 onwards):
  deep_backtracks: [
    { trigger: "term_safety_fail",
      recursion_depth: int,
      culprit_step_index: int,
      reflection: {...},
      triggering_violation: {risk_type, safety_principle, safety_bddl, ...}  <- NEW
    }, ...
  ]

A TermSafety branch is valid only if the chosen recovery does not
re-trigger the SAME BDDL rule (matched by safety_bddl text — primary
identifier; risk_type alone is insufficient because two rules in one
task can share a risk_type, e.g. top_cabinet vs bottom_cabinet
collision rules).

Legacy traces (no triggering_violation): fall back to the simple
"any other backtrack at rec_to => suppress" rule (matches old behavior;
old extracted bundles unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TermSafetyBranchInfo:
    step_index: int        # the repair_step (conversation file s-key)
    recursion_from: int    # the recursion depth that triggered the fail
    recursion_to: int      # the repair recursion depth (rec_from + 1)
    rule_id: str           # rule text (rule_id field not present; use rule text as key)
    rejected_action: str   # action from r{rec_from}_s{step_index:03d}.md
    chosen_action: str     # action from r{rec_to}_s{step_index:03d}.md (if known)
    rejected_event_index: int  # kept for schema compat (set to 0)
    chosen_event_index: int    # kept for schema compat (set to 0)
    trigger_resolved: bool


def _violation_key(v: Optional[dict]) -> str:
    """Return a string identifier for a violation dict.

    safety_bddl is the canonical unique key; falls back to
    (risk_type|safety_principle) if missing.
    """
    if not isinstance(v, dict):
        return ""
    sb_raw = v.get("safety_bddl") or v.get("bddl")  # try both key names
    sb = str(sb_raw).strip() if sb_raw else ""
    if sb:
        return sb
    rt = (v.get("risk_type") or "").strip()
    sp = (v.get("safety_principle") or "").strip()
    return f"{rt}||{sp}"


def walk_term_safety_branches(trace: dict) -> List[TermSafetyBranchInfo]:
    """Walk _trace.json and emit TermSafetyBranchInfo for resolved TS branches.

    New schema (has triggering_violation):
      A branch is emitted when the chosen repair does NOT re-trigger the SAME
      BDDL rule. If the next recursion fires a DIFFERENT rule, the chosen action
      is still considered to have resolved the original trigger, so we emit.
      If there is no next backtrack at all, the chosen resolved everything — emit.

    Legacy schema (no triggering_violation on any entry):
      A branch is emitted only when rec_to does NOT appear in the set of
      recursion depths that also had a term_safety_fail backtrack. Unchanged
      from prior behaviour so existing extracted bundles are unaffected.
    """
    deep_backtracks = trace.get("deep_backtracks") or []
    nodes = trace.get("nodes") or []

    # Build action lookup from nodes: (recursion_depth, step_index) -> last action
    node_actions: Dict[tuple, str] = {}
    for n in nodes:
        key = (n.get("recursion_depth", 0), n.get("step_index", 0))
        node_actions[key] = n.get("action", "")

    # Detect whether this is a "new schema" trace (at least one term_safety_fail
    # backtrack carries triggering_violation).
    has_new_schema = any(
        bt.get("triggering_violation")
        for bt in deep_backtracks
        if bt.get("trigger") == "term_safety_fail"
    )

    # Per-recursion-depth lookup of term_safety_fail backtrack dicts.
    backtrack_by_rec: Dict[int, dict] = {}
    for bt in deep_backtracks:
        if bt.get("trigger") != "term_safety_fail":
            continue
        backtrack_by_rec[bt.get("recursion_depth", -1)] = bt

    # Legacy fallback: set of recursion depths that fired term_safety_fail.
    failed_repair_recs: set = set(backtrack_by_rec.keys())

    out: List[TermSafetyBranchInfo] = []
    for bt in deep_backtracks:
        if bt.get("trigger") != "term_safety_fail":
            continue

        rec_from = bt.get("recursion_depth", 0)
        refl = bt.get("reflection") or {}
        repair_step = refl.get("repair_step")
        if repair_step is None:
            repair_step = bt.get("culprit_step_index")
        if repair_step is None:
            continue

        rec_to = rec_from + 1

        # ── Suppression decision ─────────────────────────────────────────────
        if has_new_schema and bt.get("triggering_violation"):
            # New schema: compare safety_bddl keys between this backtrack and
            # the next-recursion backtrack (if any).
            this_key = _violation_key(bt.get("triggering_violation"))
            next_bt = backtrack_by_rec.get(rec_to)
            if next_bt is not None:
                next_key = _violation_key(next_bt.get("triggering_violation"))
                if this_key and next_key and this_key == next_key:
                    # Exact same BDDL rule re-fired — chosen failed to resolve it.
                    continue
                # Different rules (or one is missing a key): chosen resolved this
                # trigger; the next backtrack is a new/distinct problem. Emit.
            # No next backtrack at all: chosen resolved completely. Emit.
        else:
            # Legacy path: simple "rec_to in failed set" suppress.
            if rec_to in failed_repair_recs:
                continue

        # ── Action lookup ────────────────────────────────────────────────────
        rejected_action = node_actions.get((rec_from, repair_step), "")
        chosen_action = node_actions.get((rec_to, repair_step), "")

        # Suppress if either side is missing — no meaningful pair.
        if not rejected_action or not chosen_action:
            continue

        # Rule text — prefer safety_bddl from triggering_violation when available.
        tv = bt.get("triggering_violation") or {}
        if tv:
            rule_text = (
                tv.get("safety_bddl")
                or tv.get("safety_principle")
                or refl.get("rule", "")
            )
        else:
            rule_text = refl.get("rule", "")

        out.append(TermSafetyBranchInfo(
            step_index=repair_step,
            recursion_from=rec_from,
            recursion_to=rec_to,
            rule_id=rule_text,
            rejected_action=rejected_action,
            chosen_action=chosen_action,
            rejected_event_index=0,
            chosen_event_index=0,
            trigger_resolved=True,
        ))
    return out
