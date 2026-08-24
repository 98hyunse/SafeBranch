"""Legacy → canonical migrator for ``results_dfs_*/<model>/<task>/`` trees.

Rebuilds a :class:`Sample` from the scattered legacy files:

- ``report.json``           — identity, evaluation, awareness, error_stack
- ``_trace.json``           — authoritative termination, nodes, deep_backtracks
- ``golden_trajectory.json``— optional, mirrors ``_trace.golden_trajectory``
- ``obs/r{d}_s{s}/``        — per-step image subdirs
- ``prompts/*.txt``         — last-invocation prompts (overwritten; known bug)
- ``conversations/*.md``    — per-invocation transcripts (not overwritten)

The migrator is intentionally forgiving. Partial runs (``crashed``) and
empty dirs (``never_started``) are first-class outcomes, not errors —
otherwise we re-introduce the survivorship bias that skews legacy metrics.

Ordering rule for events: ``_trace.nodes`` is already in execution order,
including rollbacks, so we iterate nodes and inject a
``BacktrackEvent(reflection_triggered)`` each time ``recursion_depth``
increases (that is what ``deep_backtracks`` encode). Node-level
``execution_rollback`` backtracks — the sibling field in ``nodes[i]`` —
are emitted inline right after the failing ``ExecuteEvent``.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Optional

from .schema import (
    ActorFailureEvent,
    Artifacts,
    BacktrackEvent,
    Config,
    CrashInfo,
    CriticEvent,
    ErrorEvent,
    Event,
    Evaluation,
    ExecuteEvent,
    ExecutionGoal,
    ExternalRefs,
    FinalStateSummary,
    HazardOrigin,
    Identity,
    ProcessSafetyCheck,
    Reflection,
    RepairPoint,
    Sample,
    Status,
    TerminalEvent,
    TerminationSafetyCheck,
    Timestamps,
)


# ---------------------------------------------------------------------------
# Mapping tables — small, explicit, and isolated so bugs show up as KeyErrors
# in tests rather than silent drift.
# ---------------------------------------------------------------------------


_TERMINATION_REASON_MAP = {
    "success": "success",
    "plan_error": "plan_error",
    "execution_error": "execution_error",
    "max_steps_exceeded": "max_steps_exceeded",
    "timeout": "timeout",
    # "done" is ambiguous: it's either retry_exhaust (see accepted_via)
    # or a plain success, resolved below.
}

_SUMMARY_TO_STATUSES = {
    "both": ("achieved", "satisfied"),
    "goal_only": ("achieved", "violated"),
    "safety_only": ("not_achieved", "satisfied"),
    "neither": ("not_achieved", "violated"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def migrate_task(task_dir: Path, model: str, runner_log: Optional[Path] = None) -> Sample:
    """Build a canonical :class:`Sample` for one legacy task directory.

    ``runner_log`` is an out-of-tree pointer (e.g. the worker log under
    ``logs/dfs_collect_<timestamp>/``) that lets ``never_started``
    samples be traced back to their crash stack.
    """

    return _Migrator(task_dir, model, runner_log).run()


def migrate_run(run_root: Path, model: str) -> list[Sample]:
    """Migrate every task directory under ``run_root``.

    ``run_root`` is the per-model folder, e.g.
    ``results_dfs_4o/Qwen3``. Subdirs whose name starts with ``_`` or
    ``.`` are ignored so stray caches don't get picked up.
    """

    samples: list[Sample] = []
    for task_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        if task_dir.name.startswith(("_", ".")):
            continue
        samples.append(migrate_task(task_dir, model))
    return samples


# ---------------------------------------------------------------------------
# Internal worker
# ---------------------------------------------------------------------------


class _Migrator:
    def __init__(self, task_dir: Path, model: str, runner_log: Optional[Path] = None):
        self.task_dir = task_dir
        self.model = model
        self.runner_log = runner_log

        self.report: Optional[dict[str, Any]] = _load_json(task_dir / "report.json")
        self.trace: Optional[dict[str, Any]] = _load_json(task_dir / "_trace.json")

        self.has_obs = (task_dir / "obs").is_dir()
        self.has_conv = (task_dir / "conversations").is_dir()
        self.has_prompts = (task_dir / "prompts").is_dir()

    # -- top-level dispatch ------------------------------------------------

    def run(self) -> Sample:
        # Identity can almost always be recovered from the dir name even
        # when every structured file is missing.
        identity = self._build_identity()
        artifacts = self._build_artifacts()
        external_refs = ExternalRefs(runner_log=str(self.runner_log) if self.runner_log else None)

        # Branch on completeness of the structured outputs.
        if not self._has_any_content():
            return Sample(
                identity=identity,
                status=Status(execution_status="never_started"),
                artifacts=artifacts,
                external_refs=external_refs,
                crash_info=CrashInfo(
                    reason="empty_task_dir",
                    partial=True,
                    evidence="task directory exists but contains no files",
                ),
            )

        if self.trace is None and self.report is None:
            return self._build_crashed_sample(identity, artifacts, external_refs)

        status = self._build_status()
        events = self._build_events()
        evaluation = self._build_evaluation()
        crash_info = self._maybe_detect_partial_crash(events)

        return Sample(
            identity=identity,
            config=self._build_config(),
            status=status,
            events=events,
            evaluation=evaluation,
            artifacts=artifacts,
            external_refs=external_refs,
            crash_info=crash_info,
            timestamps=self._build_timestamps(),
        )

    # -- identity / config -------------------------------------------------

    def _build_identity(self) -> Identity:
        task_name = self.task_dir.name
        scene = None
        model = self.model
        if self.report:
            task_name = self.report.get("task", task_name)
            scene = self.report.get("scene")
            model = self.report.get("model", model)
        elif self.trace:
            task_name = self.trace.get("task", task_name)
        return Identity(task=task_name, scene=scene or "unknown", model=model)

    def _build_config(self) -> Config:
        # Legacy files don't store full config; defaults from CLAUDE.md
        # current versions. Migration time is not the right place to
        # back-fill this, so we leave it sparse and let callers override
        # via an external config file if needed.
        return Config()

    # -- status ------------------------------------------------------------

    def _build_status(self) -> Status:
        trace = self.trace or {}
        reason = trace.get("termination_reason", "unknown")
        accepted_via = trace.get("accepted_via")

        if reason in _TERMINATION_REASON_MAP:
            execution_status = _TERMINATION_REASON_MAP[reason]
        elif reason == "done":
            execution_status = "retry_exhaust" if accepted_via == "retry_exhaust" else "success"
        else:
            execution_status = "unknown"

        final = trace.get("final_condition_status") or {}
        summary = final.get("summary")
        if summary in _SUMMARY_TO_STATUSES:
            goal_status, safety_status = _SUMMARY_TO_STATUSES[summary]
        else:
            goal_status = _derive_goal_status(final, self.report)
            safety_status = _derive_safety_status(final, self.report)

        return Status(
            execution_status=execution_status,  # type: ignore[arg-type]
            goal_status=goal_status,
            safety_status=safety_status,
            summary=summary,
        )

    # -- events ------------------------------------------------------------

    def _build_events(self) -> list[Event]:
        trace = self.trace or {}
        nodes: list[dict[str, Any]] = list(trace.get("nodes") or [])
        deep_backtracks: list[dict[str, Any]] = list(trace.get("deep_backtracks") or [])
        error_stack: list[dict[str, Any]] = list((self.report or {}).get("error_stack") or [])

        # Pre-index error_stack by action so we can attach primitive
        # failures to the matching ExecuteEvent. Multiple errors per
        # action are rare but we keep a queue just in case.
        errors_by_action: dict[str, list[dict[str, Any]]] = {}
        for err in error_stack:
            errors_by_action.setdefault(err.get("action", ""), []).append(err)

        # Group deep_backtracks by the recursion_depth they triggered.
        # Schema: a deep_backtrack at depth=d means the transition
        # d -> d+1 happened; we inject it when we see the first node at
        # depth d+1.
        backtracks_after_depth: dict[int, list[dict[str, Any]]] = {}
        for db in deep_backtracks:
            src_depth = int(db.get("recursion_depth", 0))
            backtracks_after_depth.setdefault(src_depth, []).append(db)

        events: list[Event] = []
        prev_depth: Optional[int] = None

        for node in nodes:
            depth = int(node.get("recursion_depth", 0))

            # Depth increased → emit the deep_backtrack(s) that caused it.
            if prev_depth is not None and depth > prev_depth:
                for db in backtracks_after_depth.get(prev_depth, []):
                    events.append(_deep_backtrack_to_event(db, prev_depth))

            events.extend(_node_to_events(node, errors_by_action))
            prev_depth = depth

        # Any deep_backtracks not consumed (shouldn't normally happen) get
        # appended so no information is lost.
        consumed = {id(db) for d in set(n.get("recursion_depth", 0) for n in nodes) for db in backtracks_after_depth.get(d, [])}
        for db in deep_backtracks:
            if id(db) not in consumed and prev_depth is not None and int(db.get("recursion_depth", 0)) >= prev_depth:
                events.append(_deep_backtrack_to_event(db, int(db.get("recursion_depth", 0))))

        # Remaining errors without a matching action — emit as standalone
        # ErrorEvents so the migration is lossless.
        for leftover in (e for lst in errors_by_action.values() for e in lst):
            events.append(ErrorEvent(
                error_type=leftover.get("err_type", "unknown"),
                message=leftover.get("msg", ""),
            ))

        return events

    # -- evaluation --------------------------------------------------------

    def _build_evaluation(self) -> Evaluation:
        report = self.report or {}
        trace = self.trace or {}

        execution_goal = None
        if "execution_goal_condition" in report and report["execution_goal_condition"]:
            g = report["execution_goal_condition"]
            execution_goal = ExecutionGoal(
                eval_mode=g.get("eval_mode", "bddl"),
                condition=g.get("condition", ""),
                eval=g.get("eval"),
            )

        process_safety = [
            ProcessSafetyCheck(
                action=c.get("action", ""),
                type=c.get("type", "before"),
                eval_mode=c.get("eval_mode", "bddl"),
                risk_type=c.get("risk_type", "unknown"),
                safety_principle=c.get("safety_principle", ""),
                condition=c.get("condition", ""),
                eval=c.get("eval"),
            )
            for c in report.get("process_safety_goal_condition") or []
        ]

        termination_safety = [
            TerminationSafetyCheck(
                action=c.get("action"),
                eval_mode=c.get("eval_mode", "bddl"),
                risk_type=c.get("risk_type", "unknown"),
                safety_principle=c.get("safety_principle", ""),
                condition=c.get("condition", ""),
                eval=c.get("eval"),
            )
            for c in report.get("termination_safety_goal_condition") or []
        ]

        final_state_summary = None
        if trace.get("final_condition_status"):
            fc = trace["final_condition_status"]
            final_state_summary = FinalStateSummary(
                goal_achieved=fc.get("goal_achieved"),
                term_safety_achieved=fc.get("term_safety_achieved"),
                term_violations=list(fc.get("term_violations") or []),
                raw=fc,
            )

        return Evaluation(
            execution_goal=execution_goal,
            process_safety=process_safety,
            termination_safety=termination_safety,
            awareness=None,  # legacy field is always null; reserved for future
            final_state_summary=final_state_summary,
        )

    # -- artifacts / timestamps -------------------------------------------

    def _build_artifacts(self) -> Artifacts:
        return Artifacts(
            obs_dir="obs/" if self.has_obs else None,
            prompts_dir="prompts/" if self.has_prompts else None,
            conversations_dir="conversations/" if self.has_conv else None,
            golden_trajectory_file=(
                "golden_trajectory.json"
                if (self.task_dir / "golden_trajectory.json").is_file()
                else None
            ),
        )

    def _build_timestamps(self) -> Timestamps:
        mtime = None
        for candidate in ("_trace.json", "report.json"):
            p = self.task_dir / candidate
            if p.exists():
                mtime = _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat()
                break
        return Timestamps(ended_at=mtime)

    # -- crash detection ---------------------------------------------------

    def _has_any_content(self) -> bool:
        return any(self.task_dir.iterdir())

    def _build_crashed_sample(
        self,
        identity: Identity,
        artifacts: Artifacts,
        external_refs: ExternalRefs,
    ) -> Sample:
        obs_count = _count_children(self.task_dir / "obs")
        conv_count = _count_children(self.task_dir / "conversations")
        evidence = f"obs_subdirs={obs_count}, conversations={conv_count}, no report.json/_trace.json"

        return Sample(
            identity=identity,
            status=Status(execution_status="crashed"),
            artifacts=artifacts,
            external_refs=external_refs,
            crash_info=CrashInfo(
                reason="missing_structured_output",
                partial=True,
                evidence=evidence,
            ),
        )

    def _maybe_detect_partial_crash(self, events: list[Event]) -> Optional[CrashInfo]:
        # Even if _trace.json exists, a mismatch between obs and
        # conversations can signal a write-time crash mid-step.
        if not (self.has_obs and self.has_conv):
            return None
        obs_count = _count_children(self.task_dir / "obs")
        conv_count = _count_children(self.task_dir / "conversations")
        if abs(obs_count - conv_count) > 1:
            return CrashInfo(
                reason="obs_conversations_mismatch",
                partial=True,
                evidence=f"obs_subdirs={obs_count}, conversations={conv_count}",
                last_event_index=len(events) - 1 if events else None,
            )
        return None


# ---------------------------------------------------------------------------
# Node / backtrack conversion helpers (free functions — unit-testable)
# ---------------------------------------------------------------------------


def _node_to_events(
    node: dict[str, Any],
    errors_by_action: dict[str, list[dict[str, Any]]],
) -> list[Event]:
    action = node.get("action")
    step_index = node.get("step_index")
    recursion_depth = node.get("recursion_depth")

    if action == "DONE":
        return [TerminalEvent(
            step_index=step_index,
            recursion_depth=recursion_depth,
            terminal_type="done",
        )]

    if action == "FAILED_TO_GENERATE":
        return [ActorFailureEvent(
            step_index=step_index,
            recursion_depth=recursion_depth,
            reason="FAILED_TO_GENERATE",
        )]

    bddl_before = node.get("bddl_before")
    prm_score = node.get("prm_score")

    # Pull a matching primitive-error, if any, off the queue.
    err_msg: Optional[str] = None
    if action and action in errors_by_action and errors_by_action[action]:
        err = errors_by_action[action].pop(0)
        err_msg = f"{err.get('err_type', 'error')}: {err.get('msg', '')}"

    exec_event = ExecuteEvent(
        step_index=step_index,
        recursion_depth=recursion_depth,
        action=str(action) if action is not None else "",
        bddl_before=bddl_before if bddl_before in ("pass", "fail", "fail_accepted") else None,
        prm_score=prm_score if isinstance(prm_score, int) else None,
        error=err_msg,
        success=(err_msg is None) if bddl_before == "pass" else None,
    )

    events: list[Event] = [exec_event]

    # Node-level execution rollback (Sample #4 pattern). Legacy field is
    # ``node['backtrack']`` — a string or structured dict depending on era.
    backtrack = node.get("backtrack")
    if backtrack:
        events.append(BacktrackEvent(
            step_index=step_index,
            recursion_depth=recursion_depth,
            backtrack_kind="execution_rollback",
            trigger=_stringify(backtrack) if not isinstance(backtrack, dict) else backtrack.get("trigger"),
            error=_stringify(backtrack) if not isinstance(backtrack, dict) else backtrack.get("error"),
        ))

    return events


def _deep_backtrack_to_event(db: dict[str, Any], source_depth: int) -> BacktrackEvent:
    refl_raw = db.get("reflection") or {}
    hazard = refl_raw.get("hazard_origin") or {}
    repair = refl_raw.get("repair_point") or {}

    reflection = Reflection(
        issue=refl_raw.get("issue"),
        hazard_origin=HazardOrigin(
            kind=hazard.get("kind", "unknown"),
            step_index=hazard.get("step_index"),
            explanation=hazard.get("explanation"),
        ) if hazard else None,
        repair_point=RepairPoint(
            step_index=int(repair.get("step_index", 0)),
            reason=repair.get("reason"),
        ) if repair else None,
        repair_mode=refl_raw.get("repair_mode"),
        feedback=refl_raw.get("feedback"),
        rule=refl_raw.get("rule"),
        specific_constraint=refl_raw.get("specific_constraint"),
    )

    return BacktrackEvent(
        step_index=db.get("culprit_step_index"),
        recursion_depth=source_depth,
        backtrack_kind="reflection_triggered",
        trigger=db.get("trigger"),
        culprit_step_index=db.get("culprit_step_index"),
        target_step_index=repair.get("step_index") if repair else None,
        reflection=reflection,
    )


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _count_children(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.iterdir())


def _stringify(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def _derive_goal_status(final: dict[str, Any], report: Optional[dict[str, Any]]) -> str:
    if final.get("goal_achieved") is True:
        return "achieved"
    if final.get("goal_achieved") is False:
        return "not_achieved"
    if report and isinstance(report.get("execution_goal_condition"), dict):
        ev = report["execution_goal_condition"].get("eval")
        if ev is True:
            return "achieved"
        if ev is False:
            return "not_achieved"
    return "unknown"


def _derive_safety_status(final: dict[str, Any], report: Optional[dict[str, Any]]) -> str:
    if final.get("term_safety_achieved") is True:
        return "satisfied"
    if final.get("term_safety_achieved") is False:
        return "violated"
    if not report:
        return "unknown"
    evals = []
    for check in (report.get("process_safety_goal_condition") or []) + (
        report.get("termination_safety_goal_condition") or []
    ):
        evals.append(check.get("eval"))
    if not evals or all(e is None for e in evals):
        return "unknown"
    if all(e is True for e in evals if e is not None):
        return "satisfied"
    if all(e is False for e in evals if e is not None):
        return "violated"
    return "partial"
