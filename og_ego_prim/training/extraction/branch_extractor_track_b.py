"""Track B — collect planner-emitted branch_*.json files.

Branches are stored in two sibling directories:
  - ``<task>/branches/``        — clean (confidently resolved) data
  - ``<task>/branches_review/`` — needs human inspection (chain-end, same-rule
                                   re-triggered, simulator bug suspected)

Both are collected here; downstream filtering is handled by pool_builder via
the ``validation.needs_review`` field present on every row.
"""
from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import List


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PREFIX_CACHE: dict[str, str] = {}


def _load_actor_prompt_prefix(version: str) -> str:
    """Load the actor prompt yaml's text up to (but not including) 'Your input:'.

    This is the system instruction + skills + rules + few-shot examples block
    that the planner emit hook drops on the floor. Used to backfill row's
    prompt.common at extraction time.

    The YAML uses Python `.format()` escape (`{{ }}`) so braces survive
    template substitution at DFS runtime. The actor model at eval time sees
    *unescaped* braces because the planner has already called `.format()`.
    Unescape here so training-prompt/eval-prompt distributions match.
    """
    if version in _PREFIX_CACHE:
        return _PREFIX_CACHE[version]
    yaml_path = _REPO_ROOT / "og_ego_prim" / "dfs" / "prompts" / "actor" / f"{version}.yaml"
    if not yaml_path.is_file():
        _PREFIX_CACHE[version] = ""
        return ""
    try:
        full = yaml.safe_load(yaml_path.read_text()).get("prompt", "") or ""
        idx = full.find("Your input:")
        prefix = full[:idx] if idx > 0 else full
        prefix = prefix.replace("{{", "{").replace("}}", "}")
        _PREFIX_CACHE[version] = prefix
    except Exception:
        _PREFIX_CACHE[version] = ""
    return _PREFIX_CACHE[version]


def _get_actor_version(task_dir: Path) -> str:
    meta_path = task_dir / "_run_meta.json"
    if not meta_path.is_file():
        return ""
    try:
        meta = json.loads(meta_path.read_text())
        return (meta.get("prompts") or {}).get("actor", "")
    except Exception:
        return ""


def collect_branches_from_results_dir(results_dir: Path) -> List[dict]:
    results_dir = Path(results_dir)
    rows: list[dict] = []
    # Collect from both clean and review subdirectories.
    glob_patterns = [
        "*/branches/branch_*.json",
        "*/branches_review/branch_*.json",
    ]
    for pattern in glob_patterns:
        for path in sorted(results_dir.glob(pattern)):
            try:
                row = json.loads(path.read_text())
            except json.JSONDecodeError as e:
                print(f"[track B] skip {path}: {e}")
                continue
            # source_dir is unknown to the planner; fill in here.
            row["source_dir"] = results_dir.name
            # Backfill image_paths if planner emit hook left it empty.
            # path of emitted branch json: <results_dir>/<task>/branches[_review]/branch_*.json
            # obs files: <results_dir>/<task>/obs/r{rec}_s{step:03d}/obs_*.png
            task_dir = path.parent.parent  # branches/ dir → task dir
            if not row.get("image_paths"):
                step = row.get("step_index")
                rec = row.get("recursion_from", 0)
                if isinstance(step, int):
                    obs_dir = task_dir / "obs" / f"r{rec}_s{step:03d}"
                    if obs_dir.is_dir():
                        imgs = sorted(str(p) for p in obs_dir.glob("obs_*.png"))
                        if imgs:
                            row["image_paths"] = imgs
            # Backfill prompt.common from the actor prompt yaml's prefix.
            # The planner emit hook stores only the user-input portion in
            # prompt.common (i.e. empty), losing the system instruction,
            # action vocabulary, rules, and few-shot examples. Re-attach them
            # here so downstream sft.jsonl / dpo.jsonl carry the full prompt
            # used at inference time.
            prompt = row.get("prompt") or {}
            if isinstance(prompt, dict) and not prompt.get("common"):
                actor_ver = _get_actor_version(task_dir)
                if actor_ver:
                    prefix = _load_actor_prompt_prefix(actor_ver)
                    if prefix:
                        prompt["common"] = prefix
                        row["prompt"] = prompt
            rows.append(row)
    return rows
