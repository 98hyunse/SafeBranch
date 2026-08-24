"""
Single-task DFS data-collection CLI.

Usage:
    python -m og_ego_prim.cli.dfs_collect \
        --task_name clean_the_quartz_countertop \
        --model Qwen/Qwen2.5-VL-32B-Instruct \
        --local_llm_serve \
        --local_serve_ip http://127.0.0.1:8000/v1 \
        --critic_model gpt-4o-mini \
        --output_dir results_dfs/
"""
from og_ego_prim.utils.monkey_patch import add_monkey_patch
add_monkey_patch()

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import omnigibson as og
from omnigibson.macros import gm

from og_ego_prim.benchmark import build_benchmark
from og_ego_prim.dfs.planner import DFSConfig, DFSPlanner
from og_ego_prim.models import PlanningAgent

gm.USE_GPU_DYNAMICS = True


def parse_args():
    p = argparse.ArgumentParser(description="DFS Vision-Reflection training data collector")

    # Task / scene
    p.add_argument("--task_name", type=str, required=True)
    p.add_argument("--scene_model", type=str, default=None,
                   help="Scene model name. Uses default_scene_model from task config if omitted.")

    # Actor model (Qwen)
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")
    p.add_argument("--local_llm_serve", action="store_true")
    p.add_argument("--local_serve_ip", type=str, default="http://127.0.0.1:8000/v1")
    p.add_argument("--local_serve_key", type=str, default="EMPTY")
    p.add_argument("--prompt_setting", type=str, default="v0")

    # Critic model (GPT-4o-mini)
    p.add_argument("--critic_model", type=str, default="gpt-4o-mini")

    # Prompt versions
    p.add_argument("--actor_prompt_version",       type=str, default="v0",
                   help="Version of actor/v*.yaml.")
    p.add_argument("--prm_prompt_version",         type=str, default="v0",
                   help="Version of prm/v*.yaml.")
    p.add_argument("--before_bddl_prompt_version", type=str, default="v0",
                   help="Version of before_bddl/v*.yaml.")
    p.add_argument("--task_fail_prompt_version",   type=str, default="v0",
                   help="Version of task_fail/v*.yaml.")
    p.add_argument("--term_safety_prompt_version", type=str, default="v0",
                   help="Version of term_safety/v*.yaml.")
    p.add_argument("--actor_retry_block_after_k", type=int, default=-1,
                   help="Hybrid actor v2 fallback: stop injecting the v2 retry_prompt_block "
                        "after this many same-step retries (drops prev_proposal). Lets the actor "
                        "revert to v0-equivalent prompting and exhaust into force-execute. "
                        "-1 = never fall back.")
    p.add_argument("--actor_temperature", type=float, default=0.0,
                   help="Base temperature for actor (planning) LLM calls. "
                        "The last 2 phase3 recursion depths add +0.3 on top for diversity.")

    # Actor context
    p.add_argument("--use_initial_setup", action="store_true", default=False,
                   help="Include initial scene setup description in actor prompt.")

    # DFS config
    p.add_argument("--prm_threshold", type=int, default=3,
                   help="PRM score below this triggers Phase-2 retry.")
    p.add_argument("--max_phase2_retries", type=int, default=3,
                   help="N: max retries in Phase-2 before episode termination.")
    p.add_argument("--max_before_retries", type=int, default=2,
                   help="Max same-step retries when before-BDDL check fails.")
    p.add_argument("--max_execution_retries", type=int, default=2,
                   help="Max same-step execution retries when primitive execution fails.")
    p.add_argument("--max_exec_fails_per_step", type=int, default=3,
                   help="eval_open_exec: max same-step rollback retries before forcing step advance.")
    p.add_argument("--max_phase3_recursion", type=int, default=2,
                   help="K: max deep-backtrack recursion depth.")
    p.add_argument("--carousel_threshold", type=int, default=1,
                   help="Phase3: number of past occurrences of the same (trigger, culprit_step) pair required to declare 'carousel detected'. "
                        "0 disables carousel detection (rely solely on max_phase3_recursion). "
                        "1 = current behavior (trigger on the 2nd occurrence). "
                        "N = trigger on the (N+1)-th occurrence.")
    p.add_argument("--max_steps", type=int, default=50,
                   help="Hard step limit per DFS run level.")
    p.add_argument("--step_timeout_sec", type=int, default=1200,
                   help="Per-step wall-clock cap (seconds) for executor.execute_plan "
                        "during DFS exploration and formal replay. 0 disables. "
                        "Default 1200 (20 min).")

    # Output
    p.add_argument("--output_dir", type=str, default="results_dfs")
    p.add_argument("--work_dir", type=str, default="./work_dir")

    # Misc
    p.add_argument("--draw_bbox_2d", action="store_true",
                   help="Enable 2D bbox overlay in captured observations.")
    p.add_argument("--no_prm", action="store_true",
                   help="Disable VisionPRM evaluation (skip Phase 1-a and Phase 2).")
    p.add_argument("--eval_open_exec", action="store_true", default=False,
                   help="Swallow ALL execution errors without corrective or termination (eval_open style).")
    p.add_argument("--no_before_bddl", action="store_true", default=False,
                   help="Skip Phase 1-b BeforeBDDLReflector check entirely.")
    p.add_argument("--guard_mode", type=str, default=os.environ.get("GUARD_MODE", "off"),
                   choices=["off", "gpt4o"],
                   help="RIP baseline B2 'Guard'. off=oracle BDDL per-step trigger "
                        "(default, unchanged). gpt4o=GuardClassifier replaces "
                        "the per-step unsafe trigger; unsafe routes through the existing "
                        "actor-driven before-retry recovery path. Env: GUARD_MODE.")
    p.add_argument("--guard_verifier", type=str, default=os.environ.get("GUARD_VERIFIER", "gpt4o"),
                   choices=["gpt4o", "qwen3"],
                   help="Guard verifier backend (only used when --guard_mode != off). "
                        "gpt4o=close-source GPT-4o (--critic_model + OPENAI creds). "
                        "qwen3=local Qwen3-VL vLLM at --guard_verifier_ip (RIP fairness: "
                        "same local verifier as the search baseline). Env: GUARD_VERIFIER.")
    p.add_argument("--guard_verifier_ip", type=str,
                   default=os.environ.get("GUARD_VERIFIER_IP") or os.environ.get("SERVER_IP")
                           or "http://127.0.0.1:8000/v1",
                   help="vLLM endpoint for the qwen3 guard verifier (OpenAI-compat /v1). "
                        "Defaults to the actor SERVER_IP so the verifier reuses the same "
                        "Qwen3-VL serve. Env: GUARD_VERIFIER_IP.")
    p.add_argument("--search_mode", type=str, default=os.environ.get("SEARCH_MODE", "off"),
                   choices=["off", "lookahead"],
                   help="RIP baseline B3 'Lookahead Search'. off=standard DFS "
                        "(default, unchanged). lookahead=per-step depth-1 search "
                        "(k candidates → 1-step rollout → Qwen3 SafetyValue → commit "
                        "best). Env: SEARCH_MODE.")
    p.add_argument("--search_k", type=int, default=int(os.environ.get("SEARCH_K", "3")),
                   help="Candidates expanded per lookahead decision step (default 3).")
    p.add_argument("--search_value_ip", type=str,
                   default=os.environ.get("SEARCH_VALUE_IP") or os.environ.get("SERVER_IP")
                           or "http://127.0.0.1:8000/v1",
                   help="vLLM endpoint for the Qwen3 SafetyValue (OpenAI-compat /v1). "
                        "Defaults to the actor SERVER_IP (RIP fairness: same local "
                        "verifier as the Guard). Env: SEARCH_VALUE_IP.")
    p.add_argument("--no_phase3", action="store_true", default=False,
                   help="Skip Phase 3 deep backtrack; DONE is accepted/rejected without critic repair.")
    p.add_argument("--no_task_fail_recovery", action="store_true", default=False,
                   help="BDDL goal 미달 시 task_fail deep_backtrack 스킵하고 즉시 종료. "
                        "term_safety 회복은 그대로 시도. critic-free 모드와 호환.")
    p.add_argument("--no_stall", action="store_true",
                   help="Disable the stall detector (5-step repeated action → stall_detected). Useful for ablation.")
    p.add_argument("--no_terminate_on_retry_exhaust", action="store_true", default=False,
                   help="When BeforeBDDL retry / Phase3 recursion exhausts, accept current history as golden instead of terminating.")
    p.add_argument("--emit_canonical", action="store_true", default=False,
                   help="Also write a canonical sample.json next to report.json/_trace.json "
                        "(opt-in; default behaviour is unchanged).")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _refresh_viewer_index(output_dir: str) -> None:
    """Re-export of the standalone refresh function (kept for backward compat).

    The real implementation lives in ``og_ego_prim.cli.refresh_index`` so it
    can run without importing OmniGibson (needed when the shell rebuilds the
    index after a SIGKILLed run)."""
    from og_ego_prim.cli.refresh_index import refresh_viewer_index
    refresh_viewer_index(output_dir)


def _write_task_metadata(task_name: str, task_output_dir: str) -> str | None:
    """Copy the task config (task_info + safety goals + example plan) into the
    output directory as task_metadata.json, so the viewer and DPO pair-builder
    can render goal/safety context without needing the upstream tasks/ dir.

    Also embeds a ``source_info`` provenance block (path + sha256 + mtime +
    copied_at) so downstream tooling can detect drift if the upstream file
    changes between runs.  Returns the source sha256 (or None if skipped)."""
    from og_ego_prim.utils.constants import TASKS  # upstream tasks dir default
    task_root = os.environ.get("OG_EGO_PRIM_TASKS_DIR", TASKS)
    src = os.path.join(task_root, f"{task_name}.json")
    if not os.path.exists(src):
        print(f"[dfs_collect] WARN: task config not found at {src} — skipping task_metadata.json")
        return None
    with open(src, "rb") as f:
        raw = f.read()
    sha256 = hashlib.sha256(raw).hexdigest()
    mtime_iso = datetime.fromtimestamp(os.path.getmtime(src), tz=timezone.utc).isoformat()
    cfg = json.loads(raw.decode("utf-8"))
    meta = {
        "source_info": {
            "path": os.path.relpath(src, start=os.getcwd()),
            "sha256": sha256,
            "mtime": mtime_iso,
            "copied_at": datetime.now(timezone.utc).isoformat(),
        },
        "task_info": cfg.get("task_info", {}),
        "scene_info": cfg.get("scene_info", {}),
        "planning_context": cfg.get("planning_context", {}),
        "evaluation_cautions": cfg.get("evaluation_cautions", []),
        "evaluation_goal_conditions": cfg.get("evaluation_goal_conditions", {}),
        "example_planning": cfg.get("example_planning", []),
    }
    dst = os.path.join(task_output_dir, "task_metadata.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[dfs_collect] task metadata saved → {dst} (sha {sha256[:12]})")
    return sha256


def main():
    args = parse_args()
    print(f"[dfs_collect] args: {args}")
    sys.stdout.flush()

    # ── Output directory ──────────────────────────────────────────────────────
    task_output_dir = os.path.join(args.output_dir, args.task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    report_path = os.path.join(task_output_dir, "report.json")
    trace_path  = os.path.join(task_output_dir, "_trace.json")

    # Skip if already evaluated.  Schema: execution_goal_condition lives at
    # report top-level (online_tracker.report()).  An earlier version of this
    # file looked under report['goal_condition'][...], which never matched the
    # actual schema — the skip branch was silently dead.  We now skip whenever
    # termination.reason == "done" AND execution_goal_condition.eval is a bool,
    # i.e. the task reached the goal-evaluator, regardless of pass/fail.  This
    # matches the resume use case: don't waste compute on tasks that already
    # have a real scored result; only re-run truly interrupted ones.
    if os.path.exists(report_path):
        with open(report_path) as f:
            prev = json.load(f)
        eg = prev.get("execution_goal_condition") or {}
        if prev.get("termination", {}).get("reason") == "done" and \
           isinstance(eg.get("eval"), bool):
            print(f"[dfs_collect] task already evaluated (eval={eg['eval']}) — skipping.")
            return

    # Clean stale artifacts from previous failed/incomplete runs.
    # DFS critics save prompts only once, so old prompt/image references can be misleading
    # unless we reset the prompt / obs folders for a fresh run.
    for subdir in ("obs", "prompts"):
        p = os.path.join(task_output_dir, subdir)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
    for fname in ("_trace.json", "golden_trajectory.json", "task_metadata.json"):
        p = os.path.join(task_output_dir, fname)
        if os.path.exists(p):
            os.remove(p)

    # ── Persist task metadata (goals + safety + example plan) ─────────────────
    task_source_sha = _write_task_metadata(args.task_name, task_output_dir)

    # ── Build benchmark ───────────────────────────────────────────────────────
    benchmark = build_benchmark(
        task=args.task_name,
        scene=args.scene_model,
        ego_view=True,
        draw_bbox_2d=args.draw_bbox_2d,
        use_initial_setup=False,
        use_self_caption=False,
        debug=args.debug,
        eval_process_safety=True,
        eval_termination_safety=True,
        eval_awareness=False,
        eval_execution=True,
    )

    # ── Build agent ───────────────────────────────────────────────────────────
    # PlanningAgent asserts work_dir/benchmark exists
    os.makedirs(os.path.join(args.work_dir, "benchmark"), exist_ok=True)

    agent = PlanningAgent(
        task_name=args.task_name,
        scene_name=benchmark.scene_name,
        agent_name=args.model,
        work_dir=args.work_dir,
        local_llm_serve=args.local_llm_serve,
        local_serve_ip=args.local_serve_ip,
        local_serve_key=args.local_serve_key,
        prompt_setting=args.prompt_setting,
        use_initial_setup=False,
        use_self_caption=False,
        debug=args.debug,
    )
    agent.set_tracker(benchmark.tracker)

    # ── Build DFS config ──────────────────────────────────────────────────────
    dfs_config = DFSConfig(
        prm_threshold=args.prm_threshold,
        max_phase2_retries=args.max_phase2_retries,
        max_before_retries=args.max_before_retries,
        max_execution_retries=args.max_execution_retries,
        max_exec_fails_per_step=args.max_exec_fails_per_step,
        max_phase3_recursion=args.max_phase3_recursion,
        carousel_threshold=args.carousel_threshold,
        max_steps=args.max_steps,
        step_timeout_sec=args.step_timeout_sec,
        use_prm=not args.no_prm,
        eval_open_exec=args.eval_open_exec,
        no_before_bddl=args.no_before_bddl,
        guard_mode=args.guard_mode,
        guard_verifier=args.guard_verifier,
        guard_verifier_ip=args.guard_verifier_ip,
        search_mode=args.search_mode,
        search_k=args.search_k,
        search_value_ip=args.search_value_ip,
        no_phase3=args.no_phase3,
        no_terminate_on_retry_exhaust=args.no_terminate_on_retry_exhaust,
        disable_stall_detector=args.no_stall,
        no_task_fail_recovery=args.no_task_fail_recovery,
        model_critic=args.critic_model,
        use_initial_setup=args.use_initial_setup,
        actor_prompt_version=args.actor_prompt_version,
        prm_prompt_version=args.prm_prompt_version,
        before_bddl_prompt_version=args.before_bddl_prompt_version,
        task_fail_prompt_version=args.task_fail_prompt_version,
        term_safety_prompt_version=args.term_safety_prompt_version,
        actor_retry_block_after_k=args.actor_retry_block_after_k,
        actor_temperature=args.actor_temperature,
    )

    # ── Run DFS ───────────────────────────────────────────────────────────────
    planner = DFSPlanner(
        benchmark=benchmark,
        agent=agent,
        config=dfs_config,
        output_dir=task_output_dir,
    )
    # Activate Track B branch emit: write branch_*.json under <task_dir>/branches/
    from pathlib import Path as _Path
    try:
        planner.set_branch_emit_dir(_Path(task_output_dir) / "branches")
    except Exception as _e:
        print(f"[branch emit] set_branch_emit_dir failed: {type(_e).__name__}: {_e}")

    # Embed source-config sha256 into the trace so provenance survives even
    # if task_metadata.json is later overwritten by a rerun.
    if task_source_sha is not None:
        planner.trace.task_source_sha256 = task_source_sha

    result = planner.run()

    # ── Save DFS trace + golden trajectory BEFORE formal replay ──────────────
    # OmniGibson may segfault during formal replay or cleanup.  Persisting the
    # DFS-level results first guarantees we never lose a successful trajectory.
    planner.trace.save(trace_path)
    if result.golden_trajectory:
        gt_path = os.path.join(task_output_dir, "golden_trajectory.json")
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": args.task_name,
                    "scene": benchmark.scene_name,
                    "trajectory": result.golden_trajectory,
                    "termination_reason": result.termination_reason,
                },
                f, indent=2, ensure_ascii=False,
            )
        print(f"[dfs_collect] golden trajectory saved → {gt_path}")

    # ── Formal replay (for final report metrics) ──────────────────────────────
    replay_failed = None
    try:
        planner.do_formal_replay()
    except Exception as e:
        replay_failed = f"{type(e).__name__}: {e}"
        print(f"[dfs_collect] WARN: formal replay failed ({replay_failed})")

    # ── Exploration summary (DFS-only; does NOT pollute error_stack) ──────────
    exploration_failures = getattr(planner, "_exploration_failures", []) or []
    num_unsafe_accepted = getattr(planner, "_num_unsafe_accepted", 0)
    if exploration_failures or num_unsafe_accepted:
        benchmark.tracker.exploration_summary = {
            "num_execution_failures": len(exploration_failures),
            "num_rollbacks": sum(1 for f in exploration_failures if f.get("rolled_back")),
            "num_unsafe_accepted": int(num_unsafe_accepted),
        }
    # RIP axis-2: mirror Guard inference cost into report.json (B2 baseline).
    guard_cost = getattr(planner.trace, "guard_cost", None)
    if guard_cost:
        benchmark.tracker.guard_cost = guard_cost
    # RIP axis-2: mirror Lookahead Search cost into report.json (B3 baseline).
    search_cost = getattr(planner.trace, "search_cost", None)
    if search_cost:
        benchmark.tracker.search_cost = search_cost
    if replay_failed:
        benchmark.tracker.formal_replay_error = replay_failed
        # Also mirror onto _trace.json so provenance lives independently of
        # report.json.  Wrapped so a 2nd save failure cannot mask the run.
        try:
            planner.trace.formal_replay_error = replay_failed
            planner.trace.save(trace_path)
        except Exception as e:
            print(f"[dfs_collect] WARN: trace re-save skipped ({type(e).__name__}: {e})")

    # ── Save report (after formal replay populates tracker) ───────────────────
    try:
        benchmark.tracker.save_tracking(report_path)
        print(f"[dfs_collect] report saved → {report_path}")
    except Exception as e:
        print(f"[dfs_collect] WARN: report save failed ({type(e).__name__}: {e})")

    # ── Save canonical sample (opt-in) ────────────────────────────────────────
    # Post-hoc: re-parse the legacy files we just wrote into a canonical
    # Sample. Safe because it only reads, never touches the planner.
    if args.emit_canonical:
        try:
            from og_ego_prim.harness.logger import save_canonical_for_task_dir
            canonical_path = save_canonical_for_task_dir(task_output_dir, model=args.model)
            print(f"[dfs_collect] canonical sample saved → {canonical_path}")
        except Exception as e:
            print(f"[dfs_collect] WARN: canonical emission failed ({type(e).__name__}: {e})")

    # ── Refresh viewer index (so the HTML viewer picks up this run) ──────────
    try:
        _refresh_viewer_index(args.output_dir)
    except Exception as e:
        print(f"[dfs_collect] WARN: viewer index refresh failed ({type(e).__name__}: {e})")

    print(f"\n[dfs_collect] ══ DONE  success={result.success}  "
          f"reason={result.termination_reason}  steps={len(result.golden_trajectory)} ══\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
