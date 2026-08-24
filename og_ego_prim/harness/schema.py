"""Canonical schema for IS-Bench experiment logs.

This module is the single source of truth for the log format used by the
research harness. Three clients consume it:

1. Runtime logger (``logger.py``) — writes events during live DFS runs.
2. Migration tool (``migrate.py``) — rebuilds these models from the legacy
   ``results_dfs_*/`` layout (report.json + _trace.json + obs/ + prompts/).
3. Loader + analysis (``loader.py``, Streamlit debugger, metric scripts) —
   reads logs back into memory.

Evolution rules (see SCHEMA_VERSION):
- Adding an optional field is an additive change; readers must tolerate
  unknown keys (``model_config.extra = 'allow'``).
- Adding a new event ``kind`` is additive; loaders must fall back to the
  base ``Event`` class for unknown kinds rather than crashing.
- Removing/renaming a field is a breaking change and requires bumping
  ``SCHEMA_VERSION``'s major component.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "0.1.0"


class _Base(BaseModel):
    """Base model: permissive on read (unknown fields preserved), strict on
    field types. Future additive fields won't break old loaders."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Identity, config, status
# ---------------------------------------------------------------------------


class Identity(_Base):
    task: str
    scene: str
    model: str
    task_config_hash: Optional[str] = None
    run_id: Optional[str] = None


class Config(_Base):
    """What we asked the DFS pipeline to do.

    ``prompt_versions`` mirrors the CLAUDE.md component → version table.
    ``active_critics`` is an open list; unknown names are tolerated.
    """

    prompt_versions: dict[str, str] = Field(default_factory=dict)
    active_critics: list[str] = Field(default_factory=list)
    use_prm: bool = True
    max_steps: Optional[int] = None
    timeout_s: Optional[int] = None
    extra: dict[str, Any] = Field(default_factory=dict)


ExecutionStatus = Literal[
    "success",
    "plan_error",
    "execution_error",
    "max_steps_exceeded",
    "retry_exhaust",
    "timeout",
    "crashed",
    "never_started",
    "unknown",
]

GoalStatus = Literal["achieved", "not_achieved", "unknown"]
SafetyStatus = Literal["satisfied", "violated", "partial", "unknown"]


class Status(_Base):
    """Three-axis outcome. Decoupled because IS-Bench's core finding is
    goal-achieved-but-safety-violated."""

    execution_status: ExecutionStatus = "unknown"
    goal_status: GoalStatus = "unknown"
    safety_status: SafetyStatus = "unknown"
    # Free-form summary pulled from trace (e.g. "goal_only", "neither",
    # "both"). Kept separate so structured fields stay stable.
    summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Evaluation (the scoring layer, separate from execution trace)
# ---------------------------------------------------------------------------


class ExecutionGoal(_Base):
    eval_mode: str = "bddl"
    condition: str
    eval: Optional[bool] = None  # None = not evaluated


class ProcessSafetyCheck(_Base):
    """A safety constraint that must hold **before** a specific action."""

    action: str
    type: Literal["before", "after", "during"] = "before"
    eval_mode: str = "bddl"
    risk_type: str
    safety_principle: str
    condition: str
    eval: Optional[bool] = None  # tristate: None/True/False


class TerminationSafetyCheck(_Base):
    """A safety constraint that must hold **at episode termination**."""

    action: Optional[str] = None  # the action that introduced the risk (if any)
    eval_mode: str = "bddl"
    risk_type: str
    safety_principle: str
    condition: str
    eval: Optional[bool] = None


class AwarenessRecord(_Base):
    """Optional: whether the model verbally acknowledged each risk.

    Legacy ``awareness`` field is ``None`` on most runs; structured form
    reserved for future use."""

    risk_type: str
    acknowledged: Optional[bool] = None
    citation: Optional[str] = None


class FinalStateSummary(_Base):
    """Snapshot of world at episode end. Populated by migrate.py from
    ``_trace.final_condition_status`` or similar."""

    goal_achieved: Optional[bool] = None
    term_safety_achieved: Optional[bool] = None
    term_violations: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class Evaluation(_Base):
    execution_goal: Optional[ExecutionGoal] = None
    process_safety: list[ProcessSafetyCheck] = Field(default_factory=list)
    termination_safety: list[TerminationSafetyCheck] = Field(default_factory=list)
    awareness: Optional[list[AwarenessRecord]] = None
    final_state_summary: Optional[FinalStateSummary] = None


# ---------------------------------------------------------------------------
# Events — flat list, correlated via step_index / recursion_depth / span_id
# ---------------------------------------------------------------------------
#
# Design note: we follow Inspect AI's "flat events with spans" pattern
# instead of nesting. Every event has ``step_index`` and
# ``recursion_depth`` so the DFS structure is reconstructible without
# nested data. Spans (SpanBegin/SpanEnd) group related events (e.g. all
# critic calls for one proposed action).


class _EventBase(_Base):
    kind: str  # discriminator
    step_index: Optional[int] = None
    recursion_depth: Optional[int] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    ts: Optional[float] = None  # monotonic seconds from episode start


# --- Actor / executor events ------------------------------------------------


class ExecuteEvent(_EventBase):
    """The planner proposed an action and the simulator executed it.

    ``bddl_before`` captures the legacy values: 'pass', 'fail_accepted',
    or None (e.g. for DONE markers)."""

    kind: Literal["execute"] = "execute"
    action: str
    bddl_before: Optional[Literal["pass", "fail", "fail_accepted"]] = None
    prm_score: Optional[int] = None
    success: Optional[bool] = None
    error: Optional[str] = None  # e.g. ActionPrimitiveErrorGroup stringified


class TerminalEvent(_EventBase):
    """Planner-level terminal marker: DONE, max_steps, plan_error, etc."""

    kind: Literal["terminal"] = "terminal"
    terminal_type: Literal[
        "done", "failed_to_generate", "max_steps", "timeout", "crash"
    ]
    detail: Optional[str] = None


class ActorFailureEvent(_EventBase):
    """Actor could not produce a valid next action (legacy
    ``FAILED_TO_GENERATE``)."""

    kind: Literal["actor_failure"] = "actor_failure"
    reason: Optional[str] = None
    raw_output: Optional[str] = None


# --- Critic events ----------------------------------------------------------


CriticName = Literal[
    "vision_prm",
    "before_bddl_reflector",
    "task_fail_reflector",
    "term_safety_reflector",
    "unknown",
]


class CriticEvent(_EventBase):
    """A critic was invoked. ``verdict`` summarises the decision; the full
    structured payload lives in ``payload`` for critic-specific fields."""

    kind: Literal["critic"] = "critic"
    critic: CriticName
    phase: Optional[str] = None  # "2a", "2b", "3", "4"
    verdict: Optional[Literal["pass", "block", "warn", "accept_unsafe"]] = None
    score: Optional[int] = None  # e.g. PRM 1-5
    payload: dict[str, Any] = Field(default_factory=dict)
    prompt_file: Optional[str] = None  # relative path into artifacts.prompts_dir


# --- Backtracking events ----------------------------------------------------
#
# IMPORTANT: there are TWO kinds of backtrack, currently conflated in
# legacy data. We surface them as one event type with a ``backtrack_kind``
# discriminator so analysis can reason about them uniformly.


class HazardOrigin(_Base):
    kind: Literal["context_conflict", "introduced", "unknown"]
    step_index: Optional[int] = None
    explanation: Optional[str] = None


class RepairPoint(_Base):
    step_index: int
    reason: Optional[str] = None


class Reflection(_Base):
    """Structured reflection produced by term_safety / task_fail critics.

    Mirrors ``_trace.deep_backtracks[].reflection`` but with a stable schema."""

    issue: Optional[str] = None
    hazard_origin: Optional[HazardOrigin] = None
    repair_point: Optional[RepairPoint] = None
    repair_mode: Optional[Literal["REPLAN", "INSERT", "REPLACE"]] = None
    feedback: Optional[str] = None
    rule: Optional[str] = None
    specific_constraint: Optional[str] = None


class BacktrackEvent(_EventBase):
    """Unified backtrack event.

    - ``execution_rollback``: simulator step failed, node-level rollback
      (legacy: ``nodes[].backtrack`` filled in).
    - ``reflection_triggered``: critic demanded reflection-driven repair
      (legacy: ``deep_backtracks[]`` entry).
    """

    kind: Literal["backtrack"] = "backtrack"
    backtrack_kind: Literal["execution_rollback", "reflection_triggered"]
    trigger: Optional[str] = None  # e.g. "term_safety_fail", "exec_error"
    culprit_step_index: Optional[int] = None
    target_step_index: Optional[int] = None  # where execution resumes
    reflection: Optional[Reflection] = None
    error: Optional[str] = None


# --- Unsafe-accepted (core IS-Bench failure) --------------------------------


class UnsafeAcceptedEvent(_EventBase):
    """The planner chose to proceed despite a critic flagging a hazard.

    Distinct from CriticEvent(verdict="accept_unsafe") because it marks the
    *planner decision*, not the critic output. Emitting both allows
    analysis to count 'unsafe proposed' vs 'unsafe accepted' separately."""

    kind: Literal["unsafe_accepted"] = "unsafe_accepted"
    action: str
    flagged_by: CriticName
    reason: Optional[str] = None
    hazard: Optional[str] = None


# --- Accumulated guidance (rolling memory) ----------------------------------


class GuidanceUpdateEvent(_EventBase):
    """Rolling memory (max 8) was updated. Captures the full list at this
    point so replay is possible without walking every prior event."""

    kind: Literal["guidance_update"] = "guidance_update"
    op: Literal["append", "evict", "replace", "clear"]
    entry: Optional[str] = None  # the new constraint text, if op=append
    current_guidance: list[str] = Field(default_factory=list)


# --- Span bracketing --------------------------------------------------------


class SpanBeginEvent(_EventBase):
    kind: Literal["span_begin"] = "span_begin"
    name: str  # e.g. "propose_action", "critic_round"


class SpanEndEvent(_EventBase):
    kind: Literal["span_end"] = "span_end"
    name: str
    duration_s: Optional[float] = None


# --- Generic error (infrastructure / unexpected) ----------------------------


class ErrorEvent(_EventBase):
    kind: Literal["error"] = "error"
    error_type: str
    message: str
    traceback: Optional[str] = None


# --- Discriminated union ----------------------------------------------------

Event = Annotated[
    Union[
        ExecuteEvent,
        TerminalEvent,
        ActorFailureEvent,
        CriticEvent,
        BacktrackEvent,
        UnsafeAcceptedEvent,
        GuidanceUpdateEvent,
        SpanBeginEvent,
        SpanEndEvent,
        ErrorEvent,
    ],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Artifacts, crash info, timestamps
# ---------------------------------------------------------------------------


class Artifacts(_Base):
    """Paths to heavy side files that live alongside the log, not inside it.

    All paths are relative to the sample's directory so logs remain
    portable if the run tree is moved."""

    obs_dir: Optional[str] = None          # e.g. "obs/"
    prompts_dir: Optional[str] = None      # e.g. "prompts/"
    conversations_dir: Optional[str] = None  # e.g. "conversations/"
    golden_trajectory_file: Optional[str] = None


class ExternalRefs(_Base):
    """Pointers out of the sample dir — e.g. the worker runner log that
    actually contains the crash stack for ``never_started`` samples."""

    runner_log: Optional[str] = None
    crash_stack: Optional[str] = None
    extras: dict[str, str] = Field(default_factory=dict)


class CrashInfo(_Base):
    reason: str
    last_event_index: Optional[int] = None
    partial: bool = True
    evidence: Optional[str] = None  # e.g. obs count vs conversations count


class Timestamps(_Base):
    started_at: Optional[str] = None  # ISO8601
    ended_at: Optional[str] = None
    duration_s: Optional[float] = None


# ---------------------------------------------------------------------------
# Top-level containers
# ---------------------------------------------------------------------------


class Sample(_Base):
    """One task × scene × model episode.

    This is the unit the DFS planner produces and the unit the analysis
    code consumes. Everything else (ExperimentLog) is just a collection."""

    schema_version: str = SCHEMA_VERSION
    identity: Identity
    config: Config = Field(default_factory=Config)
    status: Status = Field(default_factory=Status)
    events: list[Event] = Field(default_factory=list)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    artifacts: Artifacts = Field(default_factory=Artifacts)
    external_refs: ExternalRefs = Field(default_factory=ExternalRefs)
    crash_info: Optional[CrashInfo] = None
    timestamps: Timestamps = Field(default_factory=Timestamps)


class ExperimentLog(_Base):
    """A batch of samples produced by one CLI invocation / sweep.

    We keep this thin: the per-sample file is the source of truth, and
    this container just aggregates metadata so ``summary_*`` scripts have
    something stable to iterate over."""

    schema_version: str = SCHEMA_VERSION
    experiment_id: str
    model: str
    created_at: str  # ISO8601
    cli_args: dict[str, Any] = Field(default_factory=dict)
    samples: list[Sample] = Field(default_factory=list)
    notes: Optional[str] = None
