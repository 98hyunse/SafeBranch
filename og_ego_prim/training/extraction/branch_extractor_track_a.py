"""Track A — orchestrate parser + walker + Hindsight to emit branch rows.

For one task_dir:
  1. Walk every conversations/r{r}_s{s}.md → parse sections.
  2. Find BeforeBDDL branches (in-place, action-diff, critic OK closing).
  3. Read _trace.json → walk_term_safety_branches → resolve TermSafety
     candidates by joining (rec, step) keys with conversation files.
  4. Apply Hindsight Relabel to chosen rows.
  5. Emit dicts matching the spec §7 schema.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from og_ego_prim.training.extraction.conversation_parser import (
    Section,
    parse_conversation_md,
)
from og_ego_prim.training.extraction.hindsight_relabel import (
    relabel_chosen_input,
    strip_guidance_from_input,
)
from og_ego_prim.training.extraction.trace_walker import walk_term_safety_branches
from og_ego_prim.training.extraction.bddl_rule_matcher import pick_rule_for_branch


CONV_FILE_RE = re.compile(r"^r(\d+)_s(\d+)\.md$")
YOUR_INPUT_ANCHOR = "Your input:"
HISTORY_ANCHOR = "- history_actions:"

# Markers planner.py appends to the actor prompt after the base template when
# critic feedback is present (Step Guidance reflection, retry block). For
# training we want the *base* prompt only — strip everything from the earliest
# such marker onward so the model is not conditioned on critic hints.
GUIDANCE_MARKERS = (
    "\n\n[Step Guidance]",
    "\n\n[Previous proposal was REJECTED",
)


def _strip_guidance_block(text: str) -> str:
    cuts = [text.find(m) for m in GUIDANCE_MARKERS]
    cuts = [c for c in cuts if c >= 0]
    if not cuts:
        return text
    return text[: min(cuts)]


def split_actor_prompt(actor_input: str) -> dict:
    """Return {common, task_input, history_actions}.

    common      = everything before 'Your input:'
    task_input  = from 'Your input:' to end, with any trailing critic-feedback
                  block ([Step Guidance] / [Previous proposal was REJECTED ...])
                  stripped — training prompts must not carry critic hints, only
                  the base actor input.
    history_actions = full multi-line value after '- history_actions:',
                      terminated by the first blank line (\\n\\n) which
                      separates it from the trailing 'Return exactly...'
                      instruction block in the actor prompt template.
    """
    if not actor_input:
        return {"common": "", "task_input": "", "history_actions": ""}
    idx = actor_input.find(YOUR_INPUT_ANCHOR)
    if idx < 0:
        common, task_input = "", actor_input
    else:
        common, task_input = actor_input[:idx], actor_input[idx:]
    task_input = _strip_guidance_block(task_input)
    hist_idx = task_input.find(HISTORY_ANCHOR)
    history = ""
    if hist_idx >= 0:
        rest = task_input[hist_idx + len(HISTORY_ANCHOR):].lstrip()
        end = rest.find("\n\n")
        block = rest if end < 0 else rest[:end]
        history = block.strip()
    return {"common": common, "task_input": task_input, "history_actions": history}


def _list_conv_files(task_dir: Path) -> list[tuple[int, int, Path]]:
    out = []
    cdir = task_dir / "conversations"
    if not cdir.is_dir():
        return out
    for p in cdir.iterdir():
        m = CONV_FILE_RE.match(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), int(m.group(2)), p))
    return sorted(out)


def _find_before_bddl_pairs(sections: List[Section]) -> List[dict]:
    """Return list of {rejected_section, chosen_section, critic_section}.

    Actual observed pattern in exp10 data:
      Actor → [BDDL_BEFORE_VIOLATED marker] → BeforeBDDLReflector(critic) → Actor(with guidance)

    The marker section (BDDL_BEFORE_VIOLATED) may or may not be present between the
    rejected actor and the critic. We scan for:
      - An actor section with a valid action (rejected)
      - Followed immediately (or after one marker) by a BeforeBDDLReflector critic
        with mode != PASS
      - Followed by another actor section with a different action (chosen)

    Validation: chosen action != rejected action (critic intervention was effective).
    """
    out = []
    n = len(sections)
    i = 0
    while i < n:
        s = sections[i]
        if s.kind != "actor" or not (s.response_json or {}).get("action"):
            i += 1
            continue
        rejected_actor = s
        # Look ahead: skip at most one marker to find a BeforeBDDLReflector critic
        j = i + 1
        if j < n and sections[j].kind == "marker":
            j += 1
        if j >= n:
            i += 1
            continue
        critic_sec = sections[j]
        if not (
            critic_sec.kind == "critic"
            and "BeforeBDDL" in critic_sec.label
            and (critic_sec.response_json or {}).get("mode", "PASS") != "PASS"
        ):
            i += 1
            continue
        # Next actor after the critic is the chosen (with guidance)
        k = j + 1
        if k >= n or sections[k].kind != "actor":
            i += 1
            continue
        chosen_actor = sections[k]
        if not (chosen_actor.response_json or {}).get("action"):
            i += 1
            continue
        if rejected_actor.response_json.get("action") == chosen_actor.response_json.get("action"):
            i += 1
            continue
        out.append({
            "rejected_section": rejected_actor,
            "chosen_section": chosen_actor,
            "critic_section": critic_sec,
        })
        i = k + 1
    return out


def _branch_id(task: str, source_kind: str, step: int, rec: int) -> str:
    return f"{task}_{source_kind}_step{step}_rec{rec}"


def _build_row(
    task: str,
    source_dir: str,
    source_kind: str,
    step_index: int,
    rec_from: int,
    rec_to: int,
    rule_id: str,
    rejected_section: Section,
    chosen_section: Section,
    critic_section: Section | None,
    task_dir: Path | None = None,
    term_safety_text: str = "",
) -> dict:
    rejected_input = rejected_section.input_text or ""
    chosen_input_clean = relabel_chosen_input(
        chosen_input=chosen_section.input_text or "",
        rejected_input=rejected_input,
    )
    prompt_split = split_actor_prompt(chosen_input_clean)

    # Populate image_paths from obs PNGs for this (rec_from, step_index).
    image_paths: list[str] = []
    if task_dir is not None:
        obs_dir = Path(task_dir) / "obs" / f"r{rec_from}_s{step_index:03d}"
        if obs_dir.is_dir():
            image_paths = sorted(str(p) for p in obs_dir.glob("obs_*.png"))

    # Build critic_text for BDDL rule matching.
    if source_kind == "TermSafety":
        critic_text = term_safety_text or rule_id or ""
    else:
        cf = (critic_section.response_json or {}) if critic_section else {}
        critic_text_parts = []
        for key in ("issue", "feedback", "rule", "repair_reason", "rationale"):
            v = cf.get(key)
            if isinstance(v, str) and v:
                critic_text_parts.append(v)
        critic_text = " ".join(critic_text_parts)

    rule_meta = pick_rule_for_branch(
        task_name=task,
        source_kind=source_kind,
        critic_text=critic_text,
    )

    return {
        "branch_id": _branch_id(task, source_kind, step_index, rec_from),
        "task": task,
        "source_dir": source_dir,
        "source_kind": source_kind,
        "step_index": step_index,
        "recursion_from": rec_from,
        "recursion_to": rec_to,
        "rule_id": rule_id,
        "rule_meta": rule_meta,
        "prompt": prompt_split,
        "chosen": chosen_section.response_json or {},
        "rejected": rejected_section.response_json or {},
        "critic_feedback": (critic_section.response_json or {}) if critic_section else {},
        "validation": {
            "trigger_resolved": True,
            "validated_by": "next_critic_pass" if source_kind == "BeforeBDDL"
                            else "next_episode_end",
        },
        "image_paths": image_paths,
        "track": "A",
    }


def extract_branches_from_task_dir(task_dir: Path) -> list[dict]:
    task_dir = Path(task_dir)
    task_name = task_dir.name
    source_dir = task_dir.parent.name
    rows: list[dict] = []

    # Cache parsed sections per file.
    parsed: dict[tuple[int, int], list[Section]] = {}
    for rec, step, path in _list_conv_files(task_dir):
        parsed[(rec, step)] = parse_conversation_md(path.read_text())

    # 1. BeforeBDDL — within each conversation file.
    for (rec, step), sections in parsed.items():
        for pair in _find_before_bddl_pairs(sections):
            critic = pair["critic_section"]
            rule_id = (critic.response_json or {}).get("rule_id", "")
            row = _build_row(
                task=task_name,
                source_dir=source_dir,
                source_kind="BeforeBDDL",
                step_index=step,
                rec_from=rec,
                rec_to=rec,
                rule_id=rule_id,
                rejected_section=pair["rejected_section"],
                chosen_section=pair["chosen_section"],
                critic_section=critic,
                task_dir=task_dir,
            )
            rows.append(row)

    # 2. TermSafety — _trace.json + cross-file matching.
    trace_path = task_dir / "_trace.json"
    if trace_path.is_file():
        trace = json.loads(trace_path.read_text())
        for ts in walk_term_safety_branches(trace):
            rejected_sections = parsed.get((ts.recursion_from, ts.step_index), [])
            chosen_sections = parsed.get((ts.recursion_to, ts.step_index), [])
            rejected_actor = next(
                (s for s in rejected_sections if s.kind == "actor"), None
            )
            chosen_actor = next(
                (s for s in chosen_sections if s.kind == "actor"), None
            )
            if rejected_actor is None or chosen_actor is None:
                continue
            # Skip degenerate pairs where both actors chose the same action.
            rej_action = (rejected_actor.response_json or {}).get("action", "")
            cho_action = (chosen_actor.response_json or {}).get("action", "")
            if rej_action.strip().lower() == cho_action.strip().lower():
                continue
            row = _build_row(
                task=task_name,
                source_dir=source_dir,
                source_kind="TermSafety",
                step_index=ts.step_index,
                rec_from=ts.recursion_from,
                rec_to=ts.recursion_to,
                rule_id=ts.rule_id,
                rejected_section=rejected_actor,
                chosen_section=chosen_actor,
                critic_section=None,
                task_dir=task_dir,
                term_safety_text=ts.rule_id,
            )
            rows.append(row)

    return rows
