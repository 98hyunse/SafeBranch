"""Rebuild ``<results_dir>/_index.json`` from a results directory.

Pure stdlib (no OmniGibson / torch) so the shell can rebuild the index
after every dfs_collect.sh run — including runs that were SIGKILLed or
segfaulted mid-task and therefore never reached the in-process refresh.

Usage:
    python -m og_ego_prim.cli.refresh_index <results_dir>
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone


def refresh_viewer_index(output_dir: str) -> None:
    """Scan every subdir with a ``_trace.json`` and rewrite ``_index.json``.

    Tolerant of partial/checkpointed traces: does not require ``report.json``
    or a terminal ``termination_reason``.  Surfaces shell-level
    ``_run_status.json`` (``failed`` / ``timeout`` / ``segfault``) so the
    viewer can distinguish aborted runs from clean ones.
    """
    entries = []
    for sub in sorted(glob.glob(os.path.join(output_dir, "*/"))):
        name = os.path.basename(os.path.normpath(sub))
        if name.startswith("_") or name == "obs":
            continue
        trace_path = os.path.join(sub, "_trace.json")
        meta_path = os.path.join(sub, "task_metadata.json")
        report_path = os.path.join(sub, "report.json")
        if not os.path.exists(trace_path):
            continue
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                trace = json.load(f)
        except Exception:
            continue

        final_status = trace.get("final_condition_status") or {}
        nodes = trace.get("nodes") or []
        deep_bt = trace.get("deep_backtracks") or []
        exec_fail = trace.get("execution_failures") or []

        goal_achieved = final_status.get("goal_achieved")
        safety_achieved = final_status.get("term_safety_achieved")
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    rep = json.load(f)
                if goal_achieved is None:
                    egc = rep.get("execution_goal_condition") or {}
                    if isinstance(egc, dict) and "eval" in egc:
                        goal_achieved = bool(egc.get("eval"))
                if safety_achieved is None:
                    tsc = rep.get("termination_safety_goal_condition") or []
                    if isinstance(tsc, list) and tsc:
                        evals = [s.get("eval") for s in tsc if isinstance(s, dict) and s.get("eval") is not None]
                        safety_achieved = all(bool(e) for e in evals) if evals else None
            except Exception:
                pass

        if goal_achieved is True and safety_achieved is True:
            summary = "both_achieved"
        elif goal_achieved is True and safety_achieved is False:
            summary = "goal_only"
        elif goal_achieved is False and safety_achieved is True:
            summary = "safety_only"
        elif goal_achieved is False and safety_achieved is False:
            summary = "neither"
        else:
            summary = None

        risk_types = []
        scene_model = None
        task_instruction = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                scene_model = meta.get("scene_info", {}).get("default_scene_model")
                task_instruction = meta.get("planning_context", {}).get("task_instruction")
                for s in meta.get("evaluation_goal_conditions", {}).get("termination_safety_goal_condition", []):
                    rt = s.get("risk_type")
                    if rt and rt not in risk_types:
                        risk_types.append(rt)
                for s in meta.get("evaluation_goal_conditions", {}).get("process_safety_goal_condition", []):
                    rt = s.get("risk_type")
                    if rt and rt not in risk_types:
                        risk_types.append(rt)
            except Exception:
                pass

        run_status = None
        run_exit_code = None
        status_path = os.path.join(sub, "_run_status.json")
        if os.path.exists(status_path):
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    rs = json.load(f)
                run_status = rs.get("status")
                run_exit_code = rs.get("exit_code")
            except Exception:
                pass

        mtime = os.path.getmtime(trace_path)
        entries.append({
            "task_name": name,
            "task_instruction": task_instruction,
            "scene_model": scene_model,
            "termination_reason": trace.get("termination_reason"),
            "accepted_via": trace.get("accepted_via"),
            "run_status": run_status,
            "run_exit_code": run_exit_code,
            "goal_achieved": goal_achieved,
            "safety_achieved": safety_achieved,
            "summary": summary,
            "num_steps": len(nodes),
            "num_deep_backtracks": len(deep_bt),
            "num_execution_failures": len(exec_fail),
            "num_golden_steps": len(trace.get("golden_trajectory") or []),
            "risk_types": risk_types,
            "has_report": os.path.exists(report_path),
            "last_modified": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
        })

    index_path = os.path.join(output_dir, "_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tasks": entries,
        }, f, indent=2, ensure_ascii=False)
    print(f"[refresh_index] viewer index refreshed → {index_path} ({len(entries)} tasks)")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m og_ego_prim.cli.refresh_index <results_dir>", file=sys.stderr)
        sys.exit(2)
    results_dir = sys.argv[1]
    if not os.path.isdir(results_dir):
        print(f"[refresh_index] not a directory: {results_dir}", file=sys.stderr)
        sys.exit(2)
    refresh_viewer_index(results_dir)


if __name__ == "__main__":
    main()
