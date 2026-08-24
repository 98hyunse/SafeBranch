"""Load task BDDL safety rules and pick the best-matching rule for a critic
feedback / TermSafety reflection text.

Schema (from data/tasks/<task>.json):
  evaluation_goal_conditions:
    process_safety_goal_condition: [
      {risk_type, safety_principle, safety_tip, safety_bddl, action, type}
    ]
    termination_safety_goal_condition: [ ...same shape... ]

Matching strategy: substring overlap on safety_principle / safety_bddl /
safety_tip / action / risk_type within the critic text. Score by total
characters matched, return the rule with highest score (ties broken by
first occurrence). Returns {} if no rule scores > 0.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    # The package lives at og_ego_prim/training/extraction/. Go up 3 levels to repo root.
    return Path(__file__).resolve().parents[3]


def load_task_safety_rules(task_name: str) -> dict:
    """Returns {process: [...], termination: [...]} for a task. Empty lists
    if the task file is missing or has no rules."""
    path = _repo_root() / "data" / "tasks" / f"{task_name}.json"
    if not path.is_file():
        return {"process": [], "termination": []}
    try:
        td = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"process": [], "termination": []}
    egc = td.get("evaluation_goal_conditions") or {}
    return {
        "process": list(egc.get("process_safety_goal_condition") or []),
        "termination": list(egc.get("termination_safety_goal_condition") or []),
    }


def _score_match(rule: dict, text: str) -> int:
    """Score how much of `rule` appears as substrings in `text`. Higher = better."""
    if not isinstance(rule, dict) or not text:
        return 0
    text_lc = text.lower()
    score = 0
    # Strong signals first.
    for key in ("safety_bddl", "safety_principle"):
        v = (rule.get(key) or "").strip()
        if v and v.lower() in text_lc:
            score += len(v) * 2  # weight strong signals more
    # Weaker signals.
    for key in ("safety_tip", "action"):
        v = (rule.get(key) or "").strip()
        if v and v.lower() in text_lc:
            score += len(v)
    # risk_type is short — usually appears in many texts; lower weight.
    rt = (rule.get("risk_type") or "").strip()
    if rt and rt.lower() in text_lc:
        score += max(1, len(rt) // 2)
    return score


def best_matching_rule(
    text: str,
    rules: list,
) -> Optional[dict]:
    """Pick the rule with highest overlap with text. Returns None if all score 0."""
    best: Optional[dict] = None
    best_score = 0
    for rule in rules:
        s = _score_match(rule, text)
        if s > best_score:
            best_score = s
            best = rule
    return best


def pick_rule_for_branch(
    *,
    task_name: str,
    source_kind: str,
    critic_text: str,
) -> dict:
    """Public entry: returns a rule_meta dict suitable for branch row.

    For TermSafety branches, prefer the termination_safety list; fall back to
    process_safety. For BeforeBDDL, prefer process_safety; fall back to
    termination_safety. Returns {} if no rule matches in either list.
    """
    rules = load_task_safety_rules(task_name)
    if source_kind == "TermSafety":
        primary, fallback = rules["termination"], rules["process"]
    else:
        primary, fallback = rules["process"], rules["termination"]
    pick = best_matching_rule(critic_text, primary)
    if pick is None:
        pick = best_matching_rule(critic_text, fallback)
    if pick is None:
        return {}
    return {
        "risk_type": pick.get("risk_type", ""),
        "safety_principle": pick.get("safety_principle", ""),
        "safety_tip": pick.get("safety_tip", ""),
        "safety_bddl": pick.get("safety_bddl", ""),
    }
