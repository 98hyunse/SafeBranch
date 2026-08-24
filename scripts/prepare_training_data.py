"""Convert merged SFT/preference JSONL into the training-server format.

Reads sft.jsonl and dpo.jsonl produced by merge_branch_pools.py — these are
the *single source of truth* for which rows belong in SFT vs DPO (review row
inclusion policy + dedupe tiebreaker live in merge_branch_pools and propagate
through these two files). Two independent loops, no re-filtering.

Output schema:
  SFT row: {prompt, response, images, meta}
  DPO row: {prompt, chosen, rejected, images, meta}

Where:
  - response/chosen/rejected are JSON-stringified {action, reasoning},
    wrapped in ```json ... ``` fence to match the actor's natural emit
    distribution and what plan_agent.parse_output() expects at eval time.
  - images are absolute paths under <training_obs_root>
  - input_text dropped (garbage empty field on BeforeBDDL rows)
  - missing image_paths backfilled via same-step different-recursion fallback
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path


def resolve_task_dir(repo_root: Path, source_dir: str, task: str) -> Path | None:
    """source_dir 가 'qwen3vl32b_..._t07' 같은 results_dfs 하위든
    'results_dfs_exp10_critic_full_v2_gpt4o' 같은 repo 직속이든 둘 다 처리."""
    candidates = [
        repo_root / source_dir / task,
        repo_root / "results_dfs" / source_dir / task,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def fallback_image_paths(repo_root: Path, source_dir: str, task: str,
                         step: int, rec: int) -> list[str]:
    """Find obs/r{rec}_s{step:03d}/obs_*.png; if missing, use the closest
    recursion's same-step obs as fallback."""
    task_dir = resolve_task_dir(repo_root, source_dir, task)
    if not task_dir:
        return []
    obs_dir = task_dir / "obs"
    if not obs_dir.is_dir():
        return []

    # Direct hit
    target = obs_dir / f"r{rec}_s{step:03d}"
    if target.is_dir():
        imgs = sorted(str(p) for p in target.glob("obs_*.png"))
        if imgs:
            return imgs

    # Fallback: any recursion at same step
    same_step = sorted(p for p in obs_dir.iterdir()
                       if p.is_dir() and p.name.endswith(f"_s{step:03d}"))
    if not same_step:
        return []

    # Pick the recursion closest to the requested rec
    def _rec_of(p: Path) -> int:
        try:
            return int(p.name.split("_s")[0].lstrip("r"))
        except Exception:
            return 999

    same_step.sort(key=lambda p: abs(_rec_of(p) - rec))
    chosen_dir = same_step[0]
    return sorted(str(p) for p in chosen_dir.glob("obs_*.png"))


def to_training_paths(rel_paths: list[str], repo_root: Path, training_obs_root: Path) -> list[str]:
    """Map repository-relative observation paths onto the training machine."""
    out = []
    for p in rel_paths:
        if Path(p).is_absolute():
            # Already absolute locally. Strip repo_root before remapping.
            try:
                rel = Path(p).relative_to(repo_root)
            except ValueError:
                rel = Path(p)
            out.append(str(training_obs_root / rel))
        else:
            out.append(str(training_obs_root / p))
    return out


def clean_completion(c: dict) -> dict:
    """Drop garbage 'input_text' field (empty string on BeforeBDDL rows)."""
    if not isinstance(c, dict):
        return c
    return {k: v for k, v in c.items() if k != "input_text"}


def _fence(value: dict) -> str:
    """Serialize a {action, reasoning} dict as a ```json fence so the training
    distribution matches what the actor naturally emits and what
    plan_agent.parse_output() expects at eval time."""
    return "```json\n" + json.dumps(value, ensure_ascii=False) + "\n```"


_TASK_SCENE_CACHE: dict[str, dict] = {}


def _load_task_scene(task_name: str, task_config_dirs: list[Path]) -> dict:
    """task_name → {scene_model, room, activity_definition_id, ...}. Look across
    candidate task config dirs (default `data/tasks`; pass extras for augmented
    task sets like `data/tasks2`). Empty dict if not found. Cached.
    """
    if task_name in _TASK_SCENE_CACHE:
        return _TASK_SCENE_CACHE[task_name]
    for d in task_config_dirs:
        fp = d / f"{task_name}.json"
        if fp.is_file():
            try:
                t = json.loads(fp.read_text())
                si = t.get("scene_info") or {}
                ti = t.get("task_info") or {}
                out = {
                    "scene_model": si.get("default_scene_model", ""),
                    "scene_models": si.get("scene_models") or [],
                    "room": si.get("room", ""),
                    "task_config_dir": str(d),
                    "activity_definition_id": ti.get("activity_definition_id"),
                    "task_type": ti.get("task_type", ""),
                }
                _TASK_SCENE_CACHE[task_name] = out
                return out
            except Exception:
                pass
    _TASK_SCENE_CACHE[task_name] = {}
    return {}


def _build_meta(r: dict, task_config_dirs: list[Path]) -> dict:
    task = r.get("task") or ""
    scene = _load_task_scene(task, task_config_dirs)
    return {
        "task": task,
        "step_index": r.get("step_index"),
        "recursion_from": r.get("recursion_from"),
        "source_kind": r.get("source_kind"),
        "rule_id": r.get("rule_id"),
        "risk_type": (r.get("rule_meta") or {}).get("risk_type", ""),
        "temp_sources": r.get("temp_sources", []),
        "source_count": r.get("source_count"),
        "branch_id": r.get("branch_id"),
        "needs_review": r.get("needs_review", False),
        "scene_model": scene.get("scene_model", ""),
        "room": scene.get("room", ""),
        "task_config_dir": scene.get("task_config_dir", ""),
    }


def _resolve_images(r: dict, repo: Path, training_root: Path) -> tuple[list[str], bool]:
    """Return remapped image paths and whether fallback observations were used."""
    imgs = r.get("image_paths") or []
    used_fallback = False
    if not imgs:
        si = r.get("step_index")
        rec = r.get("recursion_from", 0)
        if isinstance(si, int):
            imgs = fallback_image_paths(repo, r.get("source_dir") or "", r.get("task") or "", si, rec)
            if imgs:
                used_fallback = True
    if not imgs:
        return [], False
    return to_training_paths(imgs, repo, training_root), used_fallback


def _prompt_str(r: dict) -> str:
    """sft.jsonl/dpo.jsonl produced by merge_branch_pools already store prompt
    as a flat string (full_prompt). Older formats stored a dict — handle both."""
    p = r.get("prompt") or ""
    if isinstance(p, str):
        return p
    parts = [p.get("common", ""), p.get("task_input", "")]
    return "\n\n".join(s for s in parts if s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="_workspace/pairs_merged",
                    help="Dir with branches.jsonl / sft.jsonl / dpo.jsonl")
    ap.add_argument("--output_dir", default="_workspace/training_data",
                    help="Output directory for training-formatted JSONL")
    ap.add_argument("--repo_root", default=".",
                    help="Repository root containing the collected result directories")
    ap.add_argument("--training_obs_root", required=True,
                    help="Absolute observation root on the training machine")
    ap.add_argument("--task_config_dir", action="append", default=None,
                    help="Where to look up `<task>.json` for scene/room metadata. "
                    "Can be passed multiple times to include augmented task sets "
                    "(e.g. --task_config_dir data/tasks --task_config_dir data/tasks2). "
                    "Default: data/tasks (relative to --repo_root). "
                    "Also reads TASK_CONFIG_DIR env if set.")
    args = ap.parse_args()

    repo = Path(args.repo_root)
    training_root = Path(args.training_obs_root)
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Task config dirs for scene/room lookup.
    import os
    if args.task_config_dir:
        task_config_dirs = [Path(d) for d in args.task_config_dir]
    else:
        env_dir = os.environ.get("TASK_CONFIG_DIR")
        task_config_dirs = [Path(env_dir)] if env_dir else [repo / "data" / "tasks"]
    for d in task_config_dirs:
        if not d.is_dir():
            print(f"warning: task config dir not found: {d} (scene meta will be empty for tasks not in other dirs)")
    print(f"task_config_dirs: {[str(d) for d in task_config_dirs]}")

    # Load sft.jsonl + dpo.jsonl produced by merge_branch_pools.py. These two
    # files already encode the review-inclusion policy (sft=clean+review,
    # dpo=clean only) and tiebreaker decisions — convert is purely a format
    # transformer, no re-filtering.
    sft_in = [json.loads(l) for l in open(in_dir / "sft.jsonl")]
    dpo_in = [json.loads(l) for l in open(in_dir / "dpo.jsonl")]
    print(f"loaded sft={len(sft_in)} (clean+review), dpo={len(dpo_in)} (clean only)")

    sft_out, dpo_out = [], []
    sft_fallback = sft_dropped = 0
    dpo_fallback = dpo_dropped = 0

    # SFT loop — input rows have `completion` (= chosen).
    for r in sft_in:
        training_images, used_fb = _resolve_images(r, repo, training_root)
        if not training_images:
            sft_dropped += 1
            continue
        if used_fb:
            sft_fallback += 1
        completion = clean_completion(r.get("completion") or {})
        sft_out.append({
            "prompt": _prompt_str(r),
            "response": _fence(completion),
            "images": training_images,
            "meta": _build_meta(r, task_config_dirs),
        })

    # DPO loop — input rows have `chosen` + `rejected`.
    for r in dpo_in:
        training_images, used_fb = _resolve_images(r, repo, training_root)
        if not training_images:
            dpo_dropped += 1
            continue
        if used_fb:
            dpo_fallback += 1
        chosen = clean_completion(r.get("chosen") or {})
        rejected = clean_completion(r.get("rejected") or {})
        dpo_out.append({
            "prompt": _prompt_str(r),
            "chosen": _fence(chosen),
            "rejected": _fence(rejected),
            "images": training_images,
            "meta": _build_meta(r, task_config_dirs),
        })

    with (out_dir / "sft_train.jsonl").open("w") as f:
        for r in sft_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "dpo_train.jsonl").open("w") as f:
        for r in dpo_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sft_review = sum(1 for r in sft_out if r["meta"].get("needs_review"))
    print(f"\n=== output ===")
    print(f"  sft_train.jsonl: {len(sft_out)} rows  (of which needs_review={sft_review})")
    print(f"  dpo_train.jsonl: {len(dpo_out)} rows")
    print(f"  fallback used: sft={sft_fallback} dpo={dpo_fallback}")
    print(f"  dropped (no image even with fallback): sft={sft_dropped} dpo={dpo_dropped}")
    print(f"  output dir: {out_dir}")

    # Sanity per-axis.
    if sft_out:
        sample = sft_out[0]
        print(f"\n=== sample ===")
        print(f"  prompt len: {len(sample['prompt'])}")
        print(f"  response len: {len(sample['response'])}")
        print(f"  images count: {len(sample['images'])}")
        print(f"  first image path: {sample['images'][0]}")
        print(f"  meta keys: {list(sample['meta'].keys())}")


if __name__ == "__main__":
    main()
