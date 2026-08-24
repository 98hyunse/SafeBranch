"""
TraceLogger — records the DFS exploration tree to a _trace.json file.

Schema (option B: existing fields preserved, DPO-facing info isolated under
``node["dpo"]`` and extra backtrack fields).  Legacy consumers (harness
migrator, summary CLI) read only the canonical fields and ignore ``dpo``.

{
  "task": str,
  "termination_reason": "success" | "max_retries_exceeded" | ...,
  "accepted_via": str | null,
  "nodes": [
    {
      # — canonical (legacy-compatible) —
      "step_index": int,
      "recursion_depth": int,
      "action": str,
      "prm_score": int | null,
      "bddl_before": "pass" | "fail" | null,
      "backtrack": null | {
        "reason": str,                          # legacy
        "original_action": str,                 # legacy
        "prm_score": int | null,                # legacy (phase-2 only)
        "reflection": dict,                     # legacy
        "retry_count": int,                     # legacy
        "new_action": str | null,               # legacy
        # — new optional fields (ignored by legacy readers) —
        "original_reasoning": str | null,
        "rendered_feedback": str | null,        # actual string prepended to actor
        "hazard_category": str | null
      },
      # — DPO-facing, isolated sub-object —
      "dpo": {
        "reasoning": str | null,                # actor reasoning at this step
        "image_paths": [str, ...],              # input obs for the step
        "guidance_snapshot": [str, ...]         # accumulated guidance at step start
      }
    },
    ...
  ],
  "deep_backtracks": [
    {
      "trigger": "task_fail" | "term_safety_fail",
      "culprit_step_index": int,
      "recursion_depth": int,
      "reflection": dict,
      # — new optional fields —
      "rendered_feedback": str | null,
      "hazard_category": str | null
    },
    ...
  ],
  "golden_trajectory": [str, ...],
  "task_metadata_ref": "task_metadata.json" | null
}
"""

import json
import os
from typing import Any, Dict, List, Optional


class TraceLogger:

    def __init__(self, task_name: str):
        self.task = task_name
        self.termination_reason: Optional[str] = None
        self.accepted_via: Optional[str] = None  # e.g. "retry_exhaust" when no_terminate_on_retry_exhaust accepts current history
        self.nodes: List[Dict[str, Any]] = []
        self.deep_backtracks: List[Dict[str, Any]] = []
        self.golden_trajectory: List[str] = []
        self.final_condition_status: Optional[Dict[str, Any]] = None
        # Top-level list of exploratory execution failures (rolled back).
        # Each entry: {step, action, error_type, error_msg, rolled_back,
        #              state_changed, before_safety}
        self.execution_failures: List[Dict[str, Any]] = []
        # Provenance: sha256 of the source data/tasks/<task>.json file that
        # seeded this run.  Duplicated from task_metadata.json so that a
        # later overwrite of task_metadata.json does not erase the link.
        self.task_source_sha256: Optional[str] = None
        # Non-fatal error from the post-run formal_replay pass, if any.
        self.formal_replay_error: Optional[str] = None
        # Run metadata — prompt versions, flags, limits, models, git, time.
        # Populated by planner after construction; rendered into _trace.json
        # and a sibling _run_meta.json on save().
        self.run_config: Optional[Dict[str, Any]] = None
        self.started_at: Optional[str] = None
        self.ended_at: Optional[str] = None
        # RIP axis-2 inference-cost accounting (e.g. GPT-4o Guard, B2 baseline).
        # Populated by the planner before save(); rendered into _trace.json and
        # _run_meta.json. None when no cost-bearing component ran.
        self.guard_cost: Optional[Dict[str, Any]] = None
        # RIP axis-2 search-cost accounting (Lookahead Search baseline, B3).
        self.search_cost: Optional[Dict[str, Any]] = None

    # ── Node logging ──────────────────────────────────────────────────────────

    def log_step(
        self,
        step_index: int,
        recursion_depth: int,
        action: str,
        prm_score: Optional[int],
        bddl_before: Optional[str],
        backtrack: Optional[Dict[str, Any]] = None,
        *,
        reasoning: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        guidance_snapshot: Optional[List[str]] = None,
    ) -> None:
        node: Dict[str, Any] = {
            "step_index": step_index,
            "recursion_depth": recursion_depth,
            "action": action,
            "prm_score": prm_score,
            "bddl_before": bddl_before,
            "backtrack": backtrack,
        }
        # DPO-facing info is isolated so legacy migrate/summary readers
        # can ignore it without any schema awareness.
        if any(x is not None for x in (reasoning, image_paths, guidance_snapshot)):
            node["dpo"] = {
                "reasoning": reasoning,
                "image_paths": list(image_paths) if image_paths else [],
                "guidance_snapshot": list(guidance_snapshot) if guidance_snapshot else [],
            }
        self.nodes.append(node)

    def log_phase2_retry(
        self,
        step_index: int,
        recursion_depth: int,
        original_action: str,
        prm_score: int,
        reflection: Dict[str, Any],
        retry_count: int,
        new_action: Optional[str],
        final_prm_score: Optional[int],
        *,
        original_reasoning: Optional[str] = None,
        rendered_feedback: Optional[str] = None,
        hazard_category: Optional[str] = None,
        reasoning: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        guidance_snapshot: Optional[List[str]] = None,
    ) -> None:
        """Log a Phase-2 retry event (replaces the rejected node with enriched info)."""
        backtrack = {
            "reason": "prm_low_score",
            "original_action": original_action,
            "prm_score": prm_score,
            "reflection": reflection,
            "retry_count": retry_count,
            "new_action": new_action,
            # — optional new fields (ignored by legacy readers) —
            "original_reasoning": original_reasoning,
            "rendered_feedback": rendered_feedback,
            "hazard_category": hazard_category,
        }
        self.log_step(
            step_index=step_index,
            recursion_depth=recursion_depth,
            action=new_action or original_action,
            prm_score=final_prm_score,
            bddl_before=None,
            backtrack=backtrack,
            reasoning=reasoning,
            image_paths=image_paths,
            guidance_snapshot=guidance_snapshot,
        )

    def log_before_retry(
        self,
        step_index: int,
        recursion_depth: int,
        original_action: str,
        reflection: Dict[str, Any],
        retry_count: int,
        new_action: Optional[str],
        prm_score: Optional[int],
        bddl_before: Optional[str],
        *,
        original_reasoning: Optional[str] = None,
        rendered_feedback: Optional[str] = None,
        hazard_category: Optional[str] = None,
        reasoning: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        guidance_snapshot: Optional[List[str]] = None,
    ) -> None:
        backtrack = {
            "reason": "bddl_before_retry",
            "original_action": original_action,
            "reflection": reflection,
            "retry_count": retry_count,
            "new_action": new_action,
            # — optional new fields —
            "original_reasoning": original_reasoning,
            "rendered_feedback": rendered_feedback,
            "hazard_category": hazard_category,
        }
        self.log_step(
            step_index=step_index,
            recursion_depth=recursion_depth,
            action=new_action or original_action,
            prm_score=prm_score,
            bddl_before=bddl_before,
            backtrack=backtrack,
            reasoning=reasoning,
            image_paths=image_paths,
            guidance_snapshot=guidance_snapshot,
        )

    def log_execution_failure(
        self,
        step_index: int,
        action: str,
        error_type: str,
        error_msg: str,
        rolled_back: bool,
        state_changed: bool,
        before_safety: Optional[bool] = None,
    ) -> None:
        """Record an exploratory execution failure (rolled back, not in final
        report).  These live only in _trace.json, never in report.json."""
        self.execution_failures.append({
            "step_index": step_index,
            "action": action,
            "error_type": error_type,
            "error_msg": error_msg,
            "rolled_back": rolled_back,
            "state_changed": state_changed,
            "before_safety": before_safety,
        })

    # ── Deep-backtrack logging ────────────────────────────────────────────────

    def log_deep_backtrack(
        self,
        trigger: str,
        culprit_step_index: int,
        recursion_depth: int,
        reflection: Dict[str, Any],
        *,
        rendered_feedback: Optional[str] = None,
        hazard_category: Optional[str] = None,
        triggering_violation: Optional[Dict[str, Any]] = None,
    ) -> None:
        event: Dict[str, Any] = {
            "trigger": trigger,
            "culprit_step_index": culprit_step_index,
            "recursion_depth": recursion_depth,
            "reflection": reflection,
            "rendered_feedback": rendered_feedback,
            "hazard_category": hazard_category,
        }
        if triggering_violation:
            event["triggering_violation"] = triggering_violation
        self.deep_backtracks.append(event)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Stamp completion time (overwrites on every save; final value reflects last write)
        from datetime import datetime, timezone
        self.ended_at = datetime.now(timezone.utc).isoformat()
        data = {
            "task": self.task,
            "task_source_sha256": self.task_source_sha256,
            "termination_reason": self.termination_reason,
            "accepted_via": self.accepted_via,
            "nodes": self.nodes,
            "deep_backtracks": self.deep_backtracks,
            "golden_trajectory": self.golden_trajectory,
            "final_condition_status": self.final_condition_status,
            "execution_failures": self.execution_failures,
            "formal_replay_error": self.formal_replay_error,
            "task_metadata_ref": "task_metadata.json",
            "run_config": self.run_config,
            "guard_cost": self.guard_cost,
            "search_cost": self.search_cost,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[dfs] trace saved → {output_path}")

        # Also dump a standalone _run_meta.json next to the trace, for fast
        # grep/scan without parsing the full trace.
        if self.run_config is not None:
            meta_path = os.path.join(
                os.path.dirname(os.path.abspath(output_path)), "_run_meta.json"
            )
            try:
                meta = {
                    "task": self.task,
                    "task_source_sha256": self.task_source_sha256,
                    "started_at": self.started_at,
                    "ended_at": self.ended_at,
                    **self.run_config,
                }
                # Mirror Guard inference cost into _run_meta.json so it can be
                # grepped/scanned alongside report.json (RIP axis-2 cost).
                if self.guard_cost is not None:
                    meta["guard_cost"] = self.guard_cost
                if self.search_cost is not None:
                    meta["search_cost"] = self.search_cost
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[dfs] warn: failed to dump _run_meta.json ({type(e).__name__}: {e})")
